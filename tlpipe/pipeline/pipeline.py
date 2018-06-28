"""
Data Analysis and Simulation Pipeline.

A data analysis pipeline is completely specified by a pipe file that specifies
both what tasks are to be run and the parameters that go to those tasks.
Included in this package are base classes for simplifying the construction of
data analysis tasks, as well as the pipeline manager which executes them.


Examples
========

Basic Tasks
-----------

A pipeline task is a subclass of :class:`TaskBase` intended to perform some small,
modular piece analysis. The developer of the task must specify what input
parameters the task expects as well as code to perform the actual processing
for the task.

Input parameters are specified by adding class attributes `params_init` which is a
dictionary whose entries are key and default value pairs. For instance a task
definition might begin with

>>> class SpamTask(TaskBase):
...     params_init = {
...                     'eggs': [],
...                   }
...
...     prefix = 'st_'

This defines a new task named :class:`SpamTask` with a parameter named *eggs*,
whose default value is an empty list.  The default values in class attribute
:attr:`SpamTask.params_init` will be updated by reading parameters with the
specified `prefix` in the pipe file, then the updated parameter dictionary will be
used as the input parameters for this task.

The actual work for the task is specified by over-ridding any of the
:meth:`~TaskBase.setup`, :meth:`~TaskBase.next` or
:meth:`~TaskBase.finish` methods (:meth:`~TaskBase.__init__` may also be
implemented`).  These are executed in order, with :meth:`~TaskBase.next`
possibly being executed many times.  Iteration of :meth:`next` is halted by
raising a :exc:`PipelineStopIteration`.  Here is a example of a somewhat
trivial but fully implemented task:

>>> class PrintEggs(TaskBase):
...
...     params_init = {
...                     'eggs': [],
...                   }
...
...     prefix = 'pe_'
...
...     def __init__(self, parameter_file_or_dict=None, feedback=2):
...
...         # Read in the parameters.
...         super(self.__class__, self).__init__(parameter_file_or_dict, feedback)
...
...         self.i = 0
...
...     def setup(self):
...         print "Setting up PrintEggs."
...
...     def next(self):
...         if self.i >= len(self.params['eggs']):
...             raise PipelineStopIteration()
...         print "Spam and %s eggs." % self.params['eggs'][self.i]
...         self.i += 1
...
...     def finish(self):
...         print "Finished PrintEggs."

Any return value of these three pipeline methods can be handled by the pipeline
and provided to subsequent tasks. The methods :meth:`setup` and :meth:`next`
may accept (positional only) arguments which will be received as the outputs of
early tasks in a pipeline chain. The following is an example of a pair of tasks
that are designed to operate in this manner.

>>> class GetEggs(TaskBase):
...
...     params_init = {
...                     'eggs': [],
...                   }
...
...     prefix = 'ge_'
...
...     def __init__(self, parameter_file_or_dict=None, feedback=2):
...
...         # Read in the parameters.
...         super(self.__class__, self).__init__(parameter_file_or_dict, feedback)
...
...         self.i = 0
...         self.eggs = self.params['eggs']
...
...     def setup(self):
...         print "Setting up GetEggs."
...
...     def next(self):
...         if self.i >= len(self.eggs):
...             raise PipelineStopIteration()
...         egg = self.eggs[self.i]
...         self.i += 1
...         return egg
...
...     def finish(self):
...         print "Finished GetEggs."


>>> class CookEggs(TaskBase):
...
...     params_init = {
...                     'style': 'fried',
...                   }
...
...     prefix = 'ce_'
...
...     def setup(self):
...         print "Setting up CookEggs."
...
...     def next(self, egg):
...         print "Cooking %s %s eggs." % (self.params['style'], egg)
...
...     def finish(self):
...         print "Finished CookEggs."

Note that :meth:`CookEggs.next` never raises a :exc:`PipelineStopIteration`.
This is because there is no way for the task to internally know how long to
iterate.  :meth:`next` will continue to be called as long as there are inputs
for :meth:`next` and will stop iterating when there are none.

Pipeline Configuration
----------------------

To actually run a task or series of tasks, a pipe pipeline configuration file
(actually is a python script) is required.  The pipeline configuration has two main
functions: to specify the the pipeline (which tasks are run, in which order and
how to handle the inputs and outputs of tasks) and to provide parameters to each
individual task.  Here is an example of a pipeline configuration:

>>> spam_pipe = '''
... pipe_tasks = []
...
... pipe_tasks.append(PrintEggs)
... ### parameters for PrintEggs
... pe_eggs = ['green', 'duck', 'ostrich']
...
... pipe_tasks.append(GetEggs)
... ### parameters for GetEggs
... ge_eggs = pe_eggs
... ge_out = 'egg'
...
... pipe_tasks.append(CookEggs)
... ### parameters for CookEggs
... ce_style = 'fried'
... ce_in = ge_out
... '''

Here the `pipe_tasks` is a list to hold a list of tasks to be executed.  Other
parameters with the specified prefix are the input parameters for the corresponding
tasks, they include three keys that all taks will have:

out
    A 'pipeline product key' or list of keys that label any return values from
    :meth:`setup`, :meth:`next` or :meth:`finish`.
requires
    A 'pipeline product key' or list of keys representing values to be passed
    as arguments to :meth:`setup`.
in
    A 'pipeline product key' or list of keys representing values to be passed
    as arguments to :meth:`next`.


Execution Order
---------------

When the above pipeline is executed it produces the following output.

>>> exec(spam_pipe)
>>> Manager(globals()).run()
Reading parameters from dictionary.
Parameters set.
parameter: logging defaulted to value: info
parameter: outdir defaulted to value: output/
Reading parameters from dictionary.
Parameters set.
parameter: out defaulted to value: None
parameter: requires defaulted to value: None
parameter: in defaulted to value: None
Reading parameters from dictionary.
Warning: Assigned an input parameter to the value of the wrong type. Parameter name: out
Parameters set.
parameter: requires defaulted to value: None
parameter: in defaulted to value: None
Reading parameters from dictionary.
Warning: Assigned an input parameter to the value of the wrong type. Parameter name: in
Parameters set.
parameter: out defaulted to value: None
parameter: requires defaulted to value: None
Setting up PrintEggs.
Setting up GetEggs.
Setting up CookEggs.
Spam and green eggs.
Cooking fried green eggs.
Spam and duck eggs.
Cooking fried duck eggs.
Spam and ostrich eggs.
Cooking fried ostrich eggs.
Finished PrintEggs.
Finished GetEggs.
Finished CookEggs.
<BLANKLINE>
<BLANKLINE>
==========================================
=                                        =
=        DONE FOR THE PIPELINE!!         =
=           CONGRATULATIONS!!            =
=                                        =
==========================================

The rules for execution order are as follows:

1. One of the methods :meth:`setup()`, :meth:`next()` or :meth:`finish()`,
   as appropriate, will be executed from each task, in order.
2. If the task method is missing its input, as specified by the 'requires' or 'in'
   keys, restart at the beginning of the `tasks` list.
3. If the input to :meth:`next()` is missing and the task is at the beginning
   of the list there will be no opportunity to generate this input. Stop iterating
   :meth:`next()` and proceed to :meth:`finish()`.
4. Once a task has executed :meth:`finish()`, remove it from the list.
5. Once a method from the last member of the `tasks` list is executed, restart
   at the beginning of the list.

If the above rules seem somewhat opaque, consider the following example which
illustrates these rules in a pipeline with a slightly more non-trivial flow.

>>> class DoNothing(TaskBase):
...
...     prefix = 'dn_'
...
...     def setup(self):
...         print "Setting up DoNothing."
...
...     def next(self, input):
...         print "DoNothing next."
...
...     def finish(self):
...         print "Finished DoNothing."

>>> new_spam_pipe = '''
... pipe_tasks = []
...
... pipe_tasks.append(GetEggs)
... ### parameters for GetEggs
... ge_eggs = pe_eggs
... ge_out = 'egg'
...
... pipe_tasks.append(CookEggs)
... ### parameters for CookEggs
... ce_style = 'fried'
... ce_in = ge_out
...
... pipe_tasks.append(DoNothing)
... ### parameters for DoNothing
... dn_in = 'non_existent_data_product'
...
... pipe_tasks.append(PrintEggs)
... ### parameters for PrintEggs
... pe_eggs = ['green', 'duck', 'ostrich']
... '''

>>> exec(new_spam_pipe)
>>> Manager(globals()).run()
Reading parameters from dictionary.
Parameters set.
parameter: logging defaulted to value: info
parameter: outdir defaulted to value: output/
Reading parameters from dictionary.
Warning: Assigned an input parameter to the value of the wrong type. Parameter name: out
Parameters set.
parameter: requires defaulted to value: None
parameter: in defaulted to value: None
Reading parameters from dictionary.
Warning: Assigned an input parameter to the value of the wrong type. Parameter name: in
Parameters set.
parameter: out defaulted to value: None
parameter: requires defaulted to value: None
Reading parameters from dictionary.
Warning: Assigned an input parameter to the value of the wrong type. Parameter name: in
Parameters set.
parameter: out defaulted to value: None
parameter: requires defaulted to value: None
Reading parameters from dictionary.
Parameters set.
parameter: out defaulted to value: None
parameter: requires defaulted to value: None
parameter: in defaulted to value: None
Setting up GetEggs.
Setting up CookEggs.
Setting up DoNothing.
Setting up PrintEggs.
Cooking fried green eggs.
Cooking fried duck eggs.
Cooking fried ostrich eggs.
Finished GetEggs.
Finished CookEggs.
Finished DoNothing.
Spam and green eggs.
Spam and duck eggs.
Spam and ostrich eggs.
Finished PrintEggs.
<BLANKLINE>
<BLANKLINE>
==========================================
=                                        =
=        DONE FOR THE PIPELINE!!         =
=           CONGRATULATIONS!!            =
=                                        =
==========================================

Notice that :meth:`DoNothing.next` is never called, since the pipeline never
generates its input, 'non_existent_data_product'.  Once everything before
:class:`DoNothing` has been executed the pipeline notices that there is no
opertunity for 'non_existent_data_product' to be generated and forces
`DoNothing` to proceed to :meth:`finish`. This also unblocks :class:`PrintEggs`
allowing it to proceed normally.

Advanced Tasks
--------------

Several subclasses of :class:`TaskBase` provide advanced functionality for tasks
that conform to the most common patterns. This functionality includes: optionally
reading inputs from disk, instead of receiving them from the pipeline;
optionally writing outputs to disk automatically; and caching the results of a
large computation to disk in an intelligent manner (not yet implemented).

Base classes providing this functionality are :class:`OneAndOne` for (at most)
one input and one output.  There are limited to a single input ('in' key) and
a single output ('out' key).  Method :meth:`~OneAndOne.process` should be
overwritten instead of :meth:`~OneAndOne.next`.  Optionally,
:meth:`~OneAndOne.read_input` and :meth:`~OneAndOne.write_output` may be
over-ridden for maximum functionality.  :meth:`~OneAndOne.setup` and
:meth:`~OneAndOne.finish` may be overridden as usual. :class:`FileIterBase`
for iterating tasks over input files.

Inheritance diagram
-------------------

.. inheritance-diagram:: TaskBase DoNothing OneAndOne FileIterBase
   :parts: 1


See the documentation for these base classes for more details.

"""


import sys
import inspect
import Queue
import collections
import logging
import os
from os import path
import warnings
import shutil
import itertools
import datetime

from caput import mpiutil
from tlpipe.kiyopy import parse_ini
from tlpipe.utils.path_util import input_path, output_path



# Set the module logger.
logger = logging.getLogger(__name__)


# Exceptions
# ----------

class PipelineConfigError(Exception):
    """Raised when there is an error setting up a pipeline."""

    pass


class PipelineRuntimeError(Exception):
    """Raised when there is a pipeline related error at runtime."""

    pass


class PipelineStopIteration(Exception):
    """This stops the iteration of `next()` in pipeline tasks.

    Pipeline tasks should raise this exception in the `next()` method to stop
    the iteration of the task and to proceed to `finish()`.

    Note that if `next()` receives input data as an argument, it is not
    required to ever raise this exception.  The pipeline will proceed to
    `finish()` once the input data has run out.

    """

    pass


class _PipelineMissingData(Exception):
    """Used for flow control when input data is yet to be produced."""

    pass


class _PipelineFinished(Exception):
    """Raised by tasks that have been completed."""

    pass


# Pipeline Manager
# ----------------

class Manager(object):
    """Pipeline manager for setting up and running pipeline tasks.

    The manager is in charge of initializing all pipeline tasks, setting them
    up by providing the appropriate parameters, then executing the methods of
    each task in the appropriate order. It also handles intermediate data
    products and ensuring that the correct products are passed between tasks.

    """

    # Define a dictionary with keys the names of parameters to be read from
    # pipeline file and values the defaults.
    params_init = {
                    'logging': 'info', # logging level
                    'tasks': [], # a list of tasks to be executed
                    'copy': True, # copy pipefile to outdir
                    'overwrite': False, # overwrite the same name pipefile if True
                    'outdir': 'output/', # output directory of pipeline data, default is current-dir/output/
                    'timing': False, # log the running time
                    'flush': False, # flush stdout buffer after each task, may slower the running
                  }

    prefix = 'pipe_'


    def __init__(self, pipefile=None, feedback=2):

        # Read in the parameters.
        self.params, self.task_params = parse_ini.parse(pipefile, self.params_init, prefix=self.prefix, return_undeclared=True, feedback=feedback)
        self.tasks = self.params['tasks']

        # timing the running
        if self.params['timing']:
            self.start_time = datetime.datetime.now()
            if mpiutil.rank0:
                print 'Start the pipeline at %s...' % self.start_time

        # set environment var
        os.environ['TL_OUTPUT'] = self.params['outdir'] + '/'

        # copy pipefile to outdir if required
        if self.params['copy']:
            base_name = path.basename(pipefile)
            dst_file = '%s/%s' % (self.params['outdir'], base_name)
            outdir = output_path(dst_file, relative=False, mkdir=True)
            if mpiutil.rank0:
                if self.params['overwrite']:
                    shutil.copy2(pipefile, dst_file)
                else:
                    if not path.exists(dst_file):
                        shutil.copy2(pipefile, dst_file)
                    else:
                        base, ext = path.splitext(dst_file)
                        for cnt in itertools.count(1):
                            dst = '%s_%d%s' % (base, cnt, ext) # add cnt to file name
                            if not path.exists(dst):
                                shutil.copy2(pipefile, dst)
                                break

            mpiutil.barrier()


    @classmethod
    def show_params(cls):
        """Show all parameters that can be set and their default values."""
        if mpiutil.rank0:

            print 'Parameters of %s:' % cls.__name__
            for key, val in cls.params_init.items():
                print '%s:  %s' % (key, val)
            print

        mpiutil.barrier()


    def run(self):
        """Main driver method for the pipeline.

        This function initializes all pipeline tasks and runs the pipeline
        through to completion.

        """

        # Set logging level.
        numeric_level = getattr(logging, self.params['logging'].upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid logging level: %s' % self.logging)
        logging.basicConfig(level=numeric_level)

        # Flush output or not
        flush = self.params['flush']

        # Initialize all tasks.
        pipeline_tasks = []
        for ii, task_spec in enumerate(self.tasks):

            # task_spec is either the TaskBase object, or a tuple, with the first
            # element being the TaskBase object and the second element being the
            # new prefix to use
            if isinstance(task_spec, tuple) :
                task =  task_spec[0]
                task.prefix = task_spec[1]
            else :
                task = task_spec

            try:
                task = self._setup_task(task)
            except PipelineConfigError as e:
                # msg = "Setting up task %d caused an error - " % ii
                msg = "Setting up task %d: %s.%s caused an error - " % (ii, task.__module__, task.__name__)
                msg += str(e)
                new_e = PipelineConfigError(msg)
                # This preserves the traceback.
                raise new_e.__class__, new_e, sys.exc_info()[2]
            pipeline_tasks.append(task)
            if mpiutil.rank0:
                logger.debug("Added %s to task list." % task.__class__.__name__)

        # flush if requires
        if flush:
            sys.stdout.flush()
            sys.stderr.flush()

        # Run the pipeline.
        while pipeline_tasks:
            for task in list(pipeline_tasks):  # Copy list so we can alter it.
                # These lines control the flow of the pipeline.
                try:
                    out = task._pipeline_next()
                except _PipelineMissingData:
                    if pipeline_tasks.index(task) == 0:
                        msg = ("%s missing input data and is at beginning of"
                               " task list. Advancing state."
                               % task.__class__.__name__)
                        if mpiutil.rank0:
                            logger.debug(msg)
                        task._pipeline_advance_state()
                    break
                except _PipelineFinished:
                    pipeline_tasks.remove(task)
                    continue
                # Now pass the output data products to any task that needs them.
                out_keys = task._out_keys
                if out is None:     # This iteration supplied no output.
                    continue
                elif len(out_keys) == 0:    # Output not handled by pipeline.
                    continue
                elif len(out_keys) == 1:
                    out = (out,)
                elif len(out_keys) != len(out):
                    msg = ('Found unexpected number of outputs in %s (got %i expected %i)' %
                           (task.__class__.__name__, len(out), len(out_keys)))
                    raise PipelineRuntimeError(msg)
                keys = str(out_keys)
                msg = "%s produced output data product with keys %s."
                msg = msg % (task.__class__.__name__, keys)
                if mpiutil.rank0:
                    logger.debug(msg)
                for receiving_task in pipeline_tasks:
                    receiving_task._pipeline_inspect_queue_product(out_keys, out)

                # flush if requires
                if flush:
                    sys.stdout.flush()
                    sys.stderr.flush()


    def _setup_task(self, task):
        """Set up a pipeline task from the spec given in the tasks list."""

        if mpiutil.rank0:
            logger.info('Initializing task: ' + str(task))

        task = task(self.task_params)

        return task

    def __del__(self):
        """Finish."""
        # remove environment var set earlier
        del(os.environ['TL_OUTPUT'])

        # get the running time
        if self.params['timing']:
            end_time = datetime.datetime.now()
            run_time = end_time - self.start_time
            days, secs, msecs = run_time.days, run_time.seconds, run_time.microseconds
            hours = secs / 3600
            mins = (secs / 60) % 60
            secs = secs % 60 + 1.0e-6 * msecs

            msg = 'Total run time:'
            if days > 0 :
                msg += (' %d day' % days)
                if days > 1:
                    msg += 's'
            if hours > 0:
                msg += (' %d hour' % hours)
                if hours > 1:
                    msg += 's'
            if mins > 0:
                msg += (' %d minute' % mins)
                if mins > 1:
                    msg += 's'
            if secs > 0:
                msg += (' %.2f seconds' % secs)

            if mpiutil.rank0:
                print
                print 'End the pipeline at %s...' % end_time
                print msg

        if mpiutil.rank0:
            # done for the pipeline
            print
            print
            print "=========================================="
            print "=                                        ="
            print "=        DONE FOR THE PIPELINE!!         ="
            print "=           CONGRATULATIONS!!            ="
            print "=                                        ="
            print "=========================================="



# Pipeline Task Base Classes
# --------------------------

class TaskBase(object):
    """Base class for all pipeline tasks.

    All pipeline tasks should inherit from this class, with functionality and
    analysis added by over-riding `__init__`, `setup`, `next` and/or
    `finish`.

    In addition, input parameters may be specified by adding adding entries
    (key and default value) in the class attribute `params`, which will then
    be updated by the corresponding parameters setttin in the input pipeline file.

    """

    params_init = {
                    'requires': None,
                    'in': None,
                    'out': None,
                    'keep_last_in': False, # if true, will only keep the last `in`
                    'timing': False, # timing the executing of next()
                  }

    prefix = 'tb_'


    # Overridable Attributes
    # -----------------------

    @classmethod
    def _get_params(cls):
        ### get all params by merging params of the all super classes

        # merge params of the all super classes
        mro = inspect.getmro(cls)
        # all_params = {}
        # use ordered dict to keep reverse mro order
        all_params = collections.OrderedDict()
        for cls in mro[-1::-1]: # reverse order
            try:
                cls_params = cls.params_init
            except AttributeError:
                continue
            all_params.update(cls_params)

        return all_params


    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # get all params that has merged params of the all super classes
        all_params = self.__class__._get_params()

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, all_params, prefix=self.prefix, feedback=feedback)

        # setup pipeline
        self._pipeline_setup()


    @classmethod
    def show_params(cls):
        """Show all parameters that can be set and their default values of this task."""
        if mpiutil.rank0:
            # get all params that has merged params of the all super classes
            all_params = cls._get_params()

            print 'Parameters of task %s:' % cls.__name__
            for key, val in all_params.items():
                print '%s:  %s' % (key, val)
            print

        mpiutil.barrier()


    def setup(self, requires=None):
        """First analysis stage of pipeline task.

        May be overridden with any number of positional only arguments
        (defaults are allowed).  Pipeline data-products will be passed as
        specified by `requires` keys in the pipeline setup.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.

        """

        pass

    def next(self, input=None):
        """Iterative analysis stage of pipeline task.

        May be overridden with any number of positional only arguments
        (defaults are allowed).  Pipeline data-products will be passed as
        specified by `in` keys in the pipeline setup.

        Function will be called repetitively until it either raises a
        `PipelineStopIteration` or, if accepting inputs, runs out of input
        data-products.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.

        """

        raise PipelineStopIteration()

    def finish(self):
        """Final analysis stage of pipeline task.

        May be overridden with no arguments.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.

        """

        pass

    @property
    def embarrassingly_parallelizable(self):
        """Override to return `True` if `next()` is trivially parallelizeable.

        This property tells the pipeline that the problem can be parallelized
        trivially. This only applies to the `next()` method, which should not
        change the state of the task.

        If this returns `True`, then the Pipeline will execute `next()` many
        times  in parallel and handle all the intermediate data efficiently.
        Otherwise `next()` must be parallelized internally if at all. `setup()`
        and `finish()` must always be parallelized internally.

        Usage of this has not implemented.

        """

        return False

    @property
    def cacheable(self):
        """Override to return `True` if caching results is implemented.

        No caching infrastructure has yet been implemented.

        """

        return False

    @property
    def history(self):
        """History that will be added to the output file."""

        hist = 'Execute %s.%s with %s.\n' % (self.__module__, self.__class__.__name__, self.params)
        if self.params.get('extra_history', '') != '':
            hist = self.params['extra_history'] + ' ' + hist

        return hist


    # Pipeline Infrastructure
    # -----------------------

    def _pipeline_setup(self):
        """Setup the 'requires', 'in' and 'out' keys for this task."""

        # Put pipeline in state such that `setup` is the next stage called.
        self._pipeline_advance_state()
        # Parse the task spec.
        requires = format_list(self.params['requires'])
        in_ = format_list(self.params['in'])
        out = format_list(self.params['out'])
        # Inspect the `setup` method to see how many arguments it takes.
        setup_argspec = inspect.getargspec(self.setup)
        # Make sure it matches `requires` keys list specified in config.
        n_requires = len(requires)
        try:
            len_defaults = len(setup_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_req = len(setup_argspec.args) - len_defaults - 1
        if n_requires < min_req:
            msg = ("Didn't get enough 'requires' keys. Expected at least"
                   " %d and only got %d" % (min_req, n_requires))
            raise PipelineConfigError(msg)
        if (n_requires > len(setup_argspec.args) - 1
            and setup_argspec.varargs is None):
            msg = ("Got too many 'requires' keys. Expected at most %d and"
                   " got %d" % (len(setup_argspec.args) - 1, n_requires))
            raise PipelineConfigError(msg)
        # Inspect the `next` method to see how many arguments it takes.
        next_argspec = inspect.getargspec(self.next)
        # Make sure it matches `in` keys list specified in config.
        n_in = len(in_)
        try:
            len_defaults = len(next_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_in = len(next_argspec.args) - len_defaults - 1
        if n_in < min_in:
            msg = ("Didn't get enough 'in' keys. Expected at least"
                   " %d and only got %d" % (min_in, n_in))
            raise PipelineConfigError(msg)
        if (n_in > len(next_argspec.args) - 1
            and next_argspec.varargs is None):
            msg = ("Got too many 'in' keys. Expected at most %d and"
                   " got %d" % (len(next_argspec.args) - 1, n_in))
            raise PipelineConfigError(msg)
        # Now that all data product keys have been verified to be valid, store
        # them on the instance.
        self._requires_keys = requires
        self._requires = [None] * n_requires
        self._in_keys = in_
        self._in = [Queue.Queue() for i in xrange(n_in)]
        self._out_keys = out

    def _pipeline_advance_state(self):
        """Advance this pipeline task to the next stage.

        The task stages are 'setup', 'next', 'finish' or 'raise'.  This
        method sets the state of the task, advancing it to the next stage.

        Also performs some clean up tasks and checks associated with changing
        stages.

        """

        if not hasattr(self, "_pipeline_state"):
            self._pipeline_state = "setup"
        elif self._pipeline_state == "setup":
            # Delete inputs to free memory.
            self._requires = None
            self._pipeline_state = "next"
        elif self._pipeline_state == "next":
            # Make sure input queues are empty then delete them so no more data
            # can be queued.
            for in_, in_key in zip(self._in, self._in_keys):
                if not in_.empty():
                    # XXX Clean up.
                    if mpiutil.rank0:
                        print "Something left: %i" % in_.qsize()

                    msg = "Task finished %s iterating `next()` but input queue \'%s\' isn't empty." % (self.__class__.__name__, in_key)
                    if mpiutil.rank0:
                        warnings.warn(msg)

            self._in = None
            self._pipeline_state = "finish"
        elif self._pipeline_state == "finish":
            self._pipeline_state = "raise"
        elif self._pipeline_state == "raise":
            pass
        else:
            raise PipelineRuntimeError()

    def _pipeline_next(self):
        """Execute the next stage of the pipeline.

        Execute `setup()`, `next()`, `finish()` or raise `PipelineFinished`
        depending on the state of the task.  Advance the state to the next
        stage if applicable.

        """

        if self._pipeline_state == "setup":
            # Check if we have all the required input data.
            for req in self._requires:
                if req is None:
                    raise _PipelineMissingData()
            else:
                msg = "Task %s calling 'setup()'." % self.__class__.__name__
                if mpiutil.rank0:
                    logger.debug(msg)
                out = self.setup(*tuple(self._requires))
                self._pipeline_advance_state()
                return out
        elif self._pipeline_state == "next":
            # Check if we have all the required input data.
            for in_ in self._in:
                if in_.empty():
                    raise _PipelineMissingData()
            else:
                # Get the next set of data to be run.
                args = ()
                for in_ in self._in:
                    args += (in_.get(),)
                try:
                    msg = "Task %s calling 'next()'." % self.__class__.__name__
                    if mpiutil.rank0:
                        logger.debug(msg)
                    if self.params['timing']:
                        stime = datetime.datetime.now()
                    out = self.next(*args)
                    if self.params['timing']:
                        etime = datetime.datetime.now()
                        if mpiutil.rank0:
                            msg = 'Executing time of %s.next(): %s [ %s - %s ]' % (self.__class__.__name__, etime - stime, stime, etime)
                            logger.info(msg)
                    return out
                except PipelineStopIteration:
                    # Finished iterating `next()`.
                    self._pipeline_advance_state()
        elif self._pipeline_state == "finish":
            msg = "Task %s calling 'finish()'." % self.__class__.__name__
            if mpiutil.rank0:
                logger.debug(msg)
            out = self.finish()
            self._pipeline_advance_state()
            return out
        elif self._pipeline_state == "raise":
            raise _PipelineFinished()
        else:
            raise PipelineRuntimeError()

    def _pipeline_inspect_queue_product(self, keys, products):
        """Inspect data products and queue them as inputs if applicable.

        Compare a list of data products keys to the keys expected by this task
        as inputs to `setup()` ('requires') and `next()` ('in').  If there is a
        match, store the corresponding data product to be used in the next
        invocation of these methods.

        """

        n_keys = len(keys)
        for ii in xrange(n_keys):
            key = keys[ii]
            product = products[ii]
            for jj, requires_key in enumerate(self._requires_keys):
                if requires_key == key:
                    # Make sure that `setup()` hasn't already been run or this
                    # data product already set.
                    msg = "%s stowing data product with key %s for 'requires'."
                    msg = msg % (self.__class__.__name__, key)
                    if mpiutil.rank0:
                        logger.debug(msg)
                    if self._requires is None:
                        msg = ("Tried to set 'requires' data product, but"
                               "`setup()` already run")
                        raise PipelineRuntimeError(msg)
                    if not self._requires[jj] is None:
                        msg = "'requires' data product set more than once"
                        raise PipelineRuntimeError(msg)
                    else:
                        # Accept the data product and store for later use.
                        self._requires[jj] = product
            for jj, in_key in enumerate(self._in_keys):
                if in_key == key:
                    msg = "%s queue data product with key %s for 'in'."
                    msg = msg % (self.__class__.__name__, key)
                    if mpiutil.rank0:
                        logger.debug(msg)
                    # Check that task is still accepting inputs.
                    if self._in is None:
                        msg = ("Tried to queue 'requires' data product, but"
                               "`next()` iteration already completed")
                        raise PipelineRuntimeError(msg)
                    else:
                        # Accept the data product and store for later use.
                        if self.params['keep_last_in']:
                            while not self._in[jj].empty():
                                self._in[jj].get()
                        self._in[jj].put(product)


class DoNothing(TaskBase):
    """Do nothing.

    This task will actually do nothing and its :meth:`next` is never called,
    since the pipeline never generates its input, 'non_existent_input'.
    As this task method is missing its input, as specified by the 'requires'
    or 'in' keys, the pipeline will restart at the beginning of the `tasks`
    list once it get to this task. Once everything before :class:`DoNothing`
    has been executed, the pipeline notices that there is no opertunity for
    'non_existent_input' to be generated and forces `DoNothing` to proceed to
    :meth:`finish`. This then unblocks its following tasks, allowing them to
    proceed normally.

    This may be useful when devise non-trivial pipeline flow.

    """

    params_init = {
                    'requires': None,
                    'in': 'non_existent_input',
                    'out': None,
                  }

    prefix = 'dn_'

    def setup(self):
        pass

    def next(self, input):
        raise RuntimeError('Something wrong happened, DoNothing.next should never be executed')

    def finish(self):
        pass


class _DryRun(object):
    pass


class OneAndOne(TaskBase):
    """Base class for tasks that have (at most) one input and one output.

    The input for the task will be preferably get from the `in` key, if no input
    product is specified by the `in` key, then input will be get from the files
    specified by the value of :attr:params[`input_files`] if it is anything other
    None, otherwise an error will be raised upon initialization.

    If the value of :attr:params[`output_files`] is anything other than None
    then the output will be written (using :meth:`write_output`) to the specified
    output files.

    """

    params_init = {
                    'copy': False, # process a copy of the input
                    'input_files': None,
                    'output_files': None,
                    'iterable': False,
                    'iter_start': 0,
                    'iter_step': 1,
                    'iter_num': None, # number of iterations
                    'dry_run_time': 0, # dry run for this number of times
                    'process_timing': False, # timing the executing of process()
                  }

    prefix = 'ob_'


    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(OneAndOne, self).__init__(parameter_file_or_dict, feedback)

        self._init_input_files()
        self._init_output_files()

        self.iterable = self.params['iterable']
        if self.iterable:
            self.iter_start = self.params['iter_start']
            self.iter_step = self.params['iter_step']
            self.iter_num = self.params['iter_num']
        self._iter_cnt = 0 # inner iter counter
        self._iter_stop = False # inner state
        self.dry_run_time = self.params['dry_run_time']

        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getargspec(self.process)
        n_args = len(pro_argspec.args) - 1
        if n_args  > 1:
            msg = ("`process` method takes more than 1 argument, which is not"
                   " allowed")
            raise PipelineConfigError(msg)
        if pro_argspec.varargs or pro_argspec.keywords or pro_argspec.defaults:
            msg = ("`process` method may not have variable length or optional"
                   " arguments")
            raise PipelineConfigError(msg)
        if n_args == 0:
            self._no_input = True
        else: # n_args == 1
            self._no_input = False

            if len(self._in) != n_args and len(self.input_files) == 0:
                msg = ("No data to iterate over. There are no 'in' keys and no 'input_files', will stop then...")
                if mpiutil.rank0:
                    logger.info(msg)
                self.stop_iteration(True)

    def _init_input_files(self):
        self.input_files = input_path(format_list(self.params['input_files']))

    def _init_output_files(self):
        self.output_files = output_path(format_list(self.params['output_files']), mkdir=False)


    @property
    def iteration(self):
        """Current iteration when `iterable` is *True*, None else."""
        if self.iterable:
            return self.iter_start + self.iter_step * self._iter_cnt
        else:
            return None

    def restart_iteration(self):
        """Re-start the iteration.

        This will re-start the iteration with the original `iter_start` and
        `iter_step` and `iter_num` unchanged.
        """
        self._iter_cnt = 0

    def stop_iteration(self, force_stop=False):
        """Determine whether to stop the iteration.

        Return *True* if this is the second iteration in the non-iterable case,
        or when run out of the given iteration numbers in the iterable case,
        or when `force\_stop` is *True*.
        """
        if not self._iter_stop:
            if force_stop:
                self._iter_stop = True
            else:
                if self.iterable:
                    # after run out of the given iter numbers
                    if self.iter_num is not None and self._iter_cnt >= self.iter_num:
                        self._iter_stop = True
                else:
                    if self._iter_cnt > 0:
                        self._iter_stop = True

        return self._iter_stop


    def next(self, input=None):
        """Should not need to override."""

        if self.stop_iteration():
            raise PipelineStopIteration()

        if self._iter_cnt < self.dry_run_time:
            self._iter_cnt += 1
            return _DryRun()

        if isinstance(input, _DryRun):
            input = None

        if input:
            if self.params['copy']:
                input = self.copy_input(input)
            input = self.cast_input(input)
        output = self.read_process_write(input)

        self._iter_cnt += 1

        return output

    def read_process_write(self, input):
        """Reads input, executes any processing and writes output."""

        # Read input if needed.
        if input is None and not self._no_input:
            if self.input_files is None or len(self.input_files) == 0:
                if mpiutil.rank0:
                    msg = 'No file to read from, will stop then...'
                    logger.info(msg)
                self.stop_iteration(True)
                return None
            if mpiutil.rank0:
                msg = "%s reading data from files:" % self.__class__.__name__
                for input_file in self.input_files:
                    msg += '\n\t%s' % input_file
                logger.info(msg)
            mpiutil.barrier()
            input = self.read_input()

        # Analyze.
        if self._no_input:
            if not input is None:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set")
            if self.params['process_timing']:
                stime = datetime.datetime.now()
            output = self.process()
            if self.params['process_timing']:
                etime = datetime.datetime.now()
                if mpiutil.rank0:
                    msg = 'Executing time of %s.process(): %s [ %s - %s ]' % (self.__class__.__name__, etime - stime, stime, etime)
                    logger.info(msg)
        else:
            if input is None:
                output = None
            else:
                if self.params['process_timing']:
                    stime = datetime.datetime.now()
                output = self.process(input)
                if self.params['process_timing']:
                    etime = datetime.datetime.now()
                    if mpiutil.rank0:
                        msg = 'Executing time of %s.process(): %s [ %s - %s ]' % (self.__class__.__name__, etime - stime, stime, etime)
                        logger.info(msg)

        # Write output if needed.
        if output is not None and len(self.output_files) != 0:
            if mpiutil.rank0:
                msg = "%s writing data to files:" % self.__class__.__name__

                # make output dirs
                for output_file in self.output_files:
                    msg += '\n\t%s' % output_file
                    output_dir = path.dirname(output_file)
                    if not path.exists(output_dir):
                        os.makedirs(output_dir)

                logger.info(msg)

            mpiutil.barrier()

            self.write_output(output)

        return output

    def read_input(self):
        """Override to implement reading inputs from disk."""

        raise NotImplementedError()

    def copy_input(self, input):
        """Override to return a copy of the input so that the original input would
        not be changed by the task."""

        return input

    def cast_input(self, input):
        """Override to support accepting pipeline inputs of various types."""

        return input

    def read_output(self, filenames):
        """Override to implement reading outputs from disk.

        Used for result caching.

        """

        raise NotImplementedError()

    def process(self, input):
        """Override this method with your data processing task.
        """

        output = input
        return output

    def write_output(self, output):
        """Override to implement writing outputs to disk."""

        raise NotImplementedError()


class FileIterBase(OneAndOne):
    """Base class for iterating tasks over input files.

    Tasks inheriting from this class should override :meth:`process` and
    optionally :meth:`setup`, :meth:`finish`, :meth:`read_input`,
    :meth:`write_output` and :meth:`cast_input`. They should not override
    :meth:`next`.

    """

    params_init = {
                    'iterable': True,
                  }

    prefix = 'fi_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(FileIterBase, self).__init__(parameter_file_or_dict, feedback)

        input_files = format_list(self.params['input_files'])
        output_files = format_list(self.params['output_files'])

        if self.iter_num is None:
            self.iter_num = (len(input_files) - self.iter_start) / self.iter_step

        if len(input_files) == 0:
            self.input_files = []
        else:
            self.input_file = input_files[self.iteration]
        if len(output_files) == 0:
            self.output_files = []
        else:
            self.output_file = output_files[self.iteration]



# # Internal Functions
# # ------------------

def format_list(val):
    """Format parameter `val` to a list."""

    if val is None:
        return []
    elif not isinstance(val, list):
        return [ val ]
    else:
        return val


def _import_class(class_path):
    """Import class dynamically from a string."""
    path_split = class_path.split('.')
    module_path = '.'.join(path_split[:-1])
    class_name = path_split[-1]
    if module_path:
        m = __import__(module_path)
        for comp in path_split[1:]:
            m = getattr(m, comp)
        task_cls = m
    else:
        task_cls = globals()[class_name]
    return task_cls


if __name__ == "__main__":
    import doctest
    doctest.testmod()

Developer's guide
=================


Write a general task
--------------------

A pipeline task is a subclass of :class:`~tlpipe.pipeline.pipeline.TaskBase`
intended to perform some small, modular piece analysis.

To write a general task, you can use the following template
:download:`general\_task.py <examples/general\_task.py>`:

   .. literalinclude:: examples/general_task.py
      :language: python
      :linenos:


The developer of the task must specify what input parameters the task
expects if it has and a prefix, as well as code to perform the actual
processing for the task.

Input parameters are specified by adding class attributes `params_init`
which is a dictionary whose entries are key and default value pairs.
A `prefix` is used to identify and read the corresponding parameters
from the input pipe file for this task.

To perform the actual processing for the task, you could first do the
necessary initialization in :meth:`__init__`, then implement three
methods :meth:`setup`, :meth:`next` and :meth:`finish`. Usually the only
necessary method to be implemented is :meth:`next`, in which the actual
processing works are done, the other methods :meth:`__init__`, :meth:`setup`,
:meth:`finish` do not need if there is no specifical initialization,
setting up, and finishing work to do. These methods are executed in order,
with :meth:`next` possibly being executed many times. Iteration of
:meth:`next` is halted by raising a :exc:`PipelineStopIteration`.

To make it work, you could put it somewhere like in `tlpipe/tlpipe/timestream/`,
and write a input pipe file like
:download:`general\_task.pipe <examples/general\_task.pipe>`:

   .. literalinclude:: examples/general_task.pipe
      :language: python
      :linenos:

then execute the task by run ::

   $ tlpipe general_task.pipe


Write a task to process timestream data
---------------------------------------

To write a task to process the timestream data (i.e., the visibility and
auxiliary data), you can use the following template
:download:`ts\_template.py <examples/ts\_template.py>`:

   .. literalinclude:: examples/ts_template.py
      :language: python
      :linenos:

Here, instead of inherit from :class:`~tlpipe.pipeline.pipeline.TaskBase`,
we inherit from its subclass
:class:`~tlpipe.timestream.timestream_task.TimestreamTask`, and implement
the method :meth:`process` (and maybe also :meth:`__init__`, :meth:`setup`,
and :meth:`finish` if necessary). The timestream data is contained in the
argument *ts*, which may be an instance of
:class:`~tlpipe.container.raw_timestream.RawTimestream` or
:class:`~tlpipe.container.timestream.Timestream`.

.. note::

   You do not need to override the method :meth:`next` now, because in the
   class :class:`~tlpipe.pipeline.pipeline.OneAndOne`, which is the super
   class of :class:`~tlpipe.timestream.timestream_task.TimestreamTask`, we
   have

   .. code-block:: python
      :emphasize-lines: 5,11

      class OneAndOne(TaskBase):

          def next(self, input=None):
              # ...
              output = self.read_process_write(input)
              # ...
              return output

          def read_process_write(self, input):
              # ...
              output = self.process(input)
              # ...
              return output


Use data operate functions in timestream tasks
----------------------------------------------

To write a task to process the timestream data, you (in most cases) only
need to implement :meth:`process` with the input timestream data contained
in its argument *ts*, as stated above. To help with the data processing, you
could use some of the data operate functions defined in the corresponding
timestream data container class, which can automatically split the data along
one axis or some axes among multiple process and iteratively process all these
data slices. For example, to write to task to process the raw timestream data
along the axis of baseline, i.e., to process a time-frequency slice of the
raw data each time, you can have the task like
:download:`ts\_task.py <examples/ts\_task.py>`:

   .. literalinclude:: examples/ts_task.py
      :language: python
      :emphasize-lines: 20
      :linenos:

To execute the task, put it somewhere like in `tlpipe/tlpipe/timestream/`,
and write a input pipe file like
:download:`ts\_task.pipe <examples/ts\_task.pipe>`:

   .. literalinclude:: examples/ts_task.pipe
      :language: python
      :linenos:

then execute the task by run ::

   $ tlpipe ts_task.pipe


These are some data operate functions that you can use:

Data operate functions of
:class:`~tlpipe.container.raw_timestream.RawTimestream` and
:class:`~tlpipe.container.timestream.Timestream`:

   .. py:class:: tlpipe.container.timestream_common.TimestreamCommon
      :noindex:

      .. py:method:: data_operate(func, op_axis=None, axis_vals=0, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: all_data_operate(func, copy_data=False, **kwargs)
         :noindex:

      .. py:method:: time_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: freq_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_and_freq_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: freq_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

Additional data operate functions of :class:`~tlpipe.container.timestream.Timestream`:

   .. py:class:: tlpipe.container.timestream.Timestream
      :noindex:

      .. py:method:: pol_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_and_pol_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: freq_and_pol_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: pol_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_freq_and_pol_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_freq_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: time_pol_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

      .. py:method:: freq_pol_and_bl_data_operate(func, full_data=False, copy_data=False, keep_dist_axis=False, **kwargs)
         :noindex:

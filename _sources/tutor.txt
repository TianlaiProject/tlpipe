Tutorial
========

.. note::

   This is intended to be a tutorial for the user of *tlpipe* package, who will
   just use the already presented tasks in the package to do some data analysis.
   For the developers of this package and those who want to do some
   developments/continuations, you may want to refer
   :doc:`guide` for a deeper introduction.


.. contents::

Prepare for the input pipe file
-------------------------------

An input pipe file is actually a *python* script file, so it follows plain
python syntax, but to emphasis that it is just used as an input pipe file
for a data analysis pipeline, usually it is named with a suffix ".pipe"
instead of ".py".

The only required argument to run a data analysis pipeline is the input pipe
file, in which one specifies all tasks to be imported and excuted, all
parameter settings for each task and also the excuting order (or flow
controlling) of the pipeline.

Here we take the waterfall plot as an example to show how to write an input
pipe file.

Non-iterative pipeline
^^^^^^^^^^^^^^^^^^^^^^

#. Create and open an file named *plot_wf.pipe* (the name can be choosen arbitrary);
#. Speicify a variable `pipe\_tasks` to hold analysis tasks that will be
   imported and excuted, and (**optionally**) a variable `pipe\_outdir` to set
   the output directory (the default value is './output/'). You can set other
   parameters related to the pipeline according to your need or just use the
   default values. All paramters and their default values can be checked by method
   :meth:`~tlpipe.pipeline.pipeline.Manager.show_params()`,
   **note**: all these parameters should be prepended with a prefix "pipe\_";

   .. literalinclude:: examples/plot_wf.pipe
      :language: python
      :lines: 1-12
      :linenos:

#. Import tasks and set task parameters:

   #. Import :class:`~tlpipe.timestream.dispatch.Dispatch` to select data to plot;

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 15-27
         :linenos:

   #. Import :class:`~tlpipe.timestream.detect_ns.Detect` to find and mask noise
      source signal;

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 29-35
         :linenos:

   #. Import :class:`~tlpipe.plot.plot_waterfall.Plot` to plot;

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 38-44
         :linenos:

The final input pipe file looks like :download:`download <examples/plot\_wf.pipe>`:

   .. literalinclude:: examples/plot_wf.pipe
      :language: python
      :emphasize-lines: 9, 22, 31, 39
      :linenos:

.. note::

   #. To show all pipeline related parameters and their default values, you
      can do:

      >>> from tlpipe.pipeline import pipeline
      >>> pipeline.Manager.prefix
      'pipe_'
      >>> pipeline.Manager.show_params()
      Parameters of Manager:
      copy:  True
      tasks:  []
      logging:  info
      flush:  False
      timing:  False
      overwrite:  False
      outdir:  output/

   #. Each imported task should be appended into the list `pipe\_tasks` in
      order to be excuted by the pipeline;
   #. Each task's paramters should be prepended with its own prefix. See the
      source file of each task to get the prefix and all paramters that can
      be set. You can also get the prefix and paramters (and their default
      values) by the following method (take :class:`~tlpipe.timestream.dispatch.Dispatch`
      for example):

      >>> from tlpipe.timestream import dispatch
      >>> dispatch.Dispatch.prefix
      'dp_'
      >>> dispatch.Dispatch.show_params()
      Parameters of task Dispatch:
      out:  None
      requires:  None
      in:  None
      iter_start:  0
      iter_step:  1
      input_files:  None
      iter_num:  None
      copy:  False
      iterable:  False
      output_files:  None
      time_select:  (0, None)
      stop:  None
      libver:  latest
      corr:  all
      exclude:  []
      check_status:  True
      dist_axis:  0
      freq_select:  (0, None)
      feed_select:  (0, None)
      tag_output_iter:  True
      tag_input_iter:  True
      start:  0
      mode:  r
      pol_select:  (0, None)
      extra_inttime:  150
      days:  1.0
      drop_days:  0.0
      exclude_bad:  True

   #. Usally the input of one task should be ether read from the data files,
      for example:

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 24
         :linenos:

      or is the output of a previously excuted task (to construct a task chain),
      for example:

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 33
         :linenos:

      .. literalinclude:: examples/plot_wf.pipe
         :language: python
         :lines: 41
         :linenos:

Iterative pipeline
^^^^^^^^^^^^^^^^^^

To make the pipeline iteratively run for several days data, or more than one
group (treat a list of files as a separate group) of data, you should set the
parameter `iterable` of each task you want to iterate to *True*, and optionally
specify an iteration number. If no iteration number is specified, the pipeline
will iteratively run until all input data has been processed. Take again the
above waterfall plot as an example, suppose you want to iteratively plot the
waterfall of 2 days data, or two separate groups of data, the input pipe file
*plot_wf_iter.pipe* :download:`download <examples/plot\_wf\_iter.pipe>` is like:

   .. literalinclude:: examples/plot_wf_iter.pipe
      :language: python
      :emphasize-lines: 20, 25, 35, 36, 46, 54
      :linenos:

.. note::

   The number of iterations can be set only once in the first task, as after
   the first task has been executed the specified number of iterations, it will
   no longer produce its output for the subsequent tasks, those task will stop
   to iterate when there is no input for it.

Non-trivial control flow
^^^^^^^^^^^^^^^^^^^^^^^^

You can run several tasks iteratively, and then run some other tasks
non-iteratively when the iterative tasks all have done.

For example, if you want the waterfall plot of two days averaged data,
you can iteratively run several tasks, each iteration for one day data, and
then combine (accumulate and average) the two days data and plot its
waterfall, just as follows shown in *plot_wf_nontrivial.pipe*
:download:`download <examples/plot\_wf\_nontrivial.pipe>`:

   .. literalinclude:: examples/plot_wf_nontrivial.pipe
      :language: python
      :emphasize-lines: 43, 77, 90
      :linenos:

.. note::

   Notice the use of the task :class:`~tlpipe.timestream.barrier.Barrier` to
   block the control flow before the executing of its subsequent tasks. As
   the task :class:`~tlpipe.timestream.barrier.Barrier` won't get its input
   from any other tasks, the pipeline will restart at the begining every time
   when it gets to execute :class:`~tlpipe.timestream.barrier.Barrier`. Once
   everything before :class:`~tlpipe.timestream.barrier.Barrier` has been
   executed, it will unblocks its subsequent tasks and allow them to proceed
   normally.

.. note::

   Note in real data analysis, the data should be RFI flagged, calibrated,
   and maybe some other processes done before the data accumulating and
   averaging, here for simplicity and easy understanding, we have omitted
   all those processes. One can refer to the real data analysis pipeline
   input files in the package's *input* directory.

Execute several times a same task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Special care need to be taken when executing several times a same task. Since
the input pipe file is just a plain python script, it will be first executed
before the parameters parsing process, the assignment of a variable will
override the same named variable before it during the excuting of the pipe
file script. So for the need of executing several times a same task, different
prefixes should be set for each of these tasks (i.e., except for the first
appeared which could have just use the default prefix of the task, all others
need to set a different prefix). To do this, you need to append a 2-tuple to
the list `pipe\_tasks`, with its first element being the imported task, and
the second element being a new prefix to use. See for example the line

   .. literalinclude:: examples/plot_wf_nontrivial.pipe
      :language: python
      :lines: 90
      :linenos:

in *plot_wf_nontrivial.pipe* in the above example.

Save intermediate data
^^^^^^^^^^^^^^^^^^^^^^

To save data that has been processed by one task (used for maybe break point
recovery, etc.), you can just set the `output\_files` paramter of this task
to be a list of file names (can only save as *hdf5* data files), then data
will be split into almost equal chunks along the time axis and save each
chunk to one of the data file. For example, see the line

   .. literalinclude:: examples/plot_wf_nontrivial.pipe
      :language: python
      :lines: 85
      :linenos:

in *plot_wf_nontrivial.pipe* in the above example.

Recovery from intermediate data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can recovery the pipeline from a break point (where you have saved the
intermediate data) by reading data from data files you have saved. To do this,
instead of set the `in` parameter, you need to set the `input\_files` paramter
to a list with elements being the saved data files. For example, see the line

   .. literalinclude:: examples/plot_wf_nontrivial.pipe
      :language: python
      :lines: 93
      :linenos:

in *plot_wf_nontrivial.pipe* in the above example.

.. note::

   If the `in` paramter and the `input\_files` parameter are both set, the
   task will get its input from the `in` paramter instead of reading data
   from the `input\_files` as it is much slower to read the data from the
   files. So in order to recovery from the break point, you should not set
   the `in` parameter, or should set `in` to be None, which is the default
   value.


Run the pipeline
----------------

Single process run
^^^^^^^^^^^^^^^^^^

If you do not have an MPI environment installed, or you just want a single
process run, just do (in case *plot_wf.pipe* is in you working directory) ::

   $ tlpipe plot_wf.pipe

or (in case *plot_wf.pipe* isn't in you working directory) ::

   $ tlpipe dir/to/plot_wf.pipe

If you want to submit and run the pipeline in the background, do like ::

   $ nohup tlpipe dir/to/plot_wf.pipe &> output.txt &

Multiple processes run
^^^^^^^^^^^^^^^^^^^^^^

To run the pipeline in parallel and distributed maner on a cluster using
multiple processes, you can do something like (in case *plot_wf.pipe* is
in you working directory) ::

   $ mpiexec -n N tlpipe plot_wf.pipe

or (in case *plot_wf.pipe* isn't in you working directory) ::

   $ mpiexec -n N tlpipe dir/to/plot_wf.pipe

If you want to submit and run the pipeline in the background on several nodes,
for example, *node2*, *node3*, *node4*, do like ::

   $ nohup mpiexec -n N -host node2,node3,node4 --map-by node tlpipe dir/to/plot_wf.pipe &> output.txt &

.. note::

   In the above commands, **N** is the number of processes you want to run!


Pipeline products and intermediate results
------------------------------------------

Pipeline products and intermediate results will be in the directory setting
by `pipe\_outdir`\ .


Other excutable commands
------------------------

* *h5info*: Check what's in a (or a list of) HDF5 data file(s).
  For its use, do some thing like ::

     $ h5info data.hdf5

  or ::

     $ h5info data1.hdf5, data2.hdf5, data3.hdf5

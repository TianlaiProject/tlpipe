Tutorial
========

.. note::

   This is intended to be a tutorial for the user of *tlpipe* package, who will
   just use the already presented tasks in the package to do some data analysis.
   For the developers of this package and those who want to do some
   developments/continuations, you may want to refer
   :doc:`guide` for a deeper introduction.


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

#. Create and open an file named *plot_wf.pipe* (the name can be choosen arbitrary);
#. Speicify a variable `pipe\_tasks` to hold analysis tasks that will be
   imported and excuted, and (**optionally**) a variable `pipe\_outdir` to set
   the output directory (the default value is './output/'). You can set other
   parameters related to the pipeline according to your need or just use the
   default values. All paramters and their default values can be checked by method
   :meth:`~tlpipe.pipeline.pipeline.Manager.show_params()`,
   **note**: all these parameters should be prepended with a prefix "pipe\_";

   .. literalinclude:: plot_wf.pipe
      :language: python
      :lines: 1-12
      :linenos:

#. Import tasks and set task parameters:

   #. Import :class:`~tlpipe.timestream.dispatch.Dispatch` to select data to plot;

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 15-27
         :linenos:

   #. Import :class:`~tlpipe.timestream.detect_ns.Detect` to find and mask noise
      source signal;

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 29-35
         :linenos:

   #. Import :class:`~tlpipe.plot.plot_waterfall.Plot` to plot;

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 38-44
         :linenos:

The final input pipe file looks like :download:`download <plot_wf.pipe>`:

   .. literalinclude:: plot_wf.pipe
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
      exclude_bad:  True

   #. Usally the input of one task should be ether read from the data files,
      for example:

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 24
         :linenos:

      or is the output of a previously excuted task (to construct a task chain),
      for example:

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 33
         :linenos:

      .. literalinclude:: plot_wf.pipe
         :language: python
         :lines: 41
         :linenos:


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

Multiple process run
^^^^^^^^^^^^^^^^^^^^

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

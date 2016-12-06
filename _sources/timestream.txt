.. tlpipe.timestream:

.. currentmodule:: tlpipe.timestream

:mod:`timestream` -- Timestream data containers and operating tasks
===================================================================

Data containers
---------------

.. autosummary::
   :toctree: generated/

   container
   timestream_common
   raw_timestream
   timestream

Timestream base task
--------------------

.. autosummary::
   :toctree: generated/

   tod_task

Operating tasks
---------------

.. autosummary::
   :toctree: generated/

   dispatch
   detect_ns
   line_rfi
   time_flag
   freq_flag
   ns_cal
   rt2ts
   ps_fit
   ps_cal
   phs2src
   phs2zen
   phase_closure
   ps_subtract
   daytime_mask
   re_order
   accumulate
   barrier
   average
   freq_rebin
   map_making

Utilities
-----------------

.. autosummary::
   :toctree: generated/

   sg_filter
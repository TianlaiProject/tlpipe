Introduction
============

- This is a Python project for the Tianlai data processing pipeline.
- This software can simply run as a single process on a single compute node,
  but for higher performance, it can also use the Message Passing Interface
  (MPI) to run data processing tasks distributed and parallelly on multiple
  computie nodes / supercomputers.
- It mainly focuses on the Tianlai cylinder array's data processing, though its
  basic framework and many tasks also work for Tianlai dish array's data, it
  does not specifically tuned for that currently.
- It can fulfill data processing tasks from reading data from raw observation
  data files, to RFI flagging, to relative and absolute calibration, to
  map-making, etc. It also provides some plotting tasks for data visualization.
- Currrently, foreground subtractiong and power spectrum estimation have not
  been implemented, but they will come maybe in the near future.
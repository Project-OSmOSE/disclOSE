🐳 Usage
========

.. _usage:

**disclOSE** turns APLOSE-formatted detection files (automatic detections or manual annotations)
into a :class:`~disclose.dataclass.detection_data.DetectionData` object,
which wraps a pandas ``DataFrame`` and exposes filtering, reshaping, metrics and plots.

The typical workflow is:

1. describe the file and the filters in a :class:`~disclose.dataclass.detection_config.DetectionConfig`;
2. load it into a :class:`~disclose.dataclass.detection_data.DetectionData`;
3. adjust it (timezone, bounds, annotator/label selection);
4. compute metrics or plot it through ``data.plot``.

The notebooks in :doc:`examples` show this workflow on a small dataset.

Input format
------------

Detection files are **APLOSE** `results <https://project-osmose.github.io/APLOSE/user/annotation-campaign/phase-progress-result/>`_ files.

Creating a ``DetectionData``
----------------------------

The shortest way is to pass a dictionary of :class:`~disclose.dataclass.detection_config.DetectionConfig` fields:

.. code-block:: python

    from pathlib import Path

    from pandas import Timedelta

    from disclose.dataclass.detection_data import DetectionData

    config = {
        "detection_file": Path("APOCADO_results.csv"),
        "filename_format": "%Y-%m-%dT%H-%M-%S",
        "timebin_new": Timedelta("1min"),
    }

    data = DetectionData.from_dict(config)
    print(data)  # start/end datetime, annotators, labels and dataset

You can also build the configuration explicitly:

.. code-block:: python

    from disclose.dataclass.detection_config import DetectionConfig

    config = DetectionConfig(detection_file=Path("APOCADO_results.csv"))
    data = DetectionData.from_config(config, coordinates=(48.39, -4.49))

The most common configuration fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Effect
   * - ``detection_file``
     - ``Path``
     - Path to the APLOSE result file.
   * - ``filename_format``
     - ``str``
     - ``strptime`` format of the timestamps in the audio file names. Needed to reshape the time bin.
   * - ``timebin_new``
     - ``Timedelta``
     - New time bin of the detections (see :ref:`usage_timebin`).
   * - ``start_datetime`` / ``end_datetime``
     - Timestamp
     - Keep only the detections in this period.
   * - ``annotator`` / ``label``
     - ``str`` | ``list[str]``
     - Keep only these annotators / labels.
   * - ``min_frequency`` / ``max_frequency``
     - float
     - Keep only detections within this frequency range.
   * - ``confidence``
     - ``float``
     - Minimum confidence threshold.
   * - ``type``
     - ``"WEAK"`` | ``"BOX"``
     - Set to ``"WEAK"`` to keep only the weak detections.
   * - ``recording_file``
     - ``Path``
     - Recording periods file, used to compute the :ref:`recording effort <usage_effort>`.

Several files
~~~~~~~~~~~~~

``DetectionData.from_dict`` also accepts a list of configurations.
By default the resulting instances are merged into one
(see :meth:`~disclose.dataclass.detection_data.DetectionData.merge`);
pass ``merge=False`` to get the list of instances instead.
If the files do not share the same timezone, all the timestamps are converted to UTC.

.. code-block:: python

    data = DetectionData.from_dict([config_a, config_b])
    datasets = DetectionData.from_dict([config_a, config_b], merge=False)

.. _usage_timebin:

Reshaping the time bin
----------------------

Setting ``timebin_new`` in the configuration reshapes the detections to a new time bin when loading
(this needs ``filename_format`` to find the start of each audio file).
The original time bin is stored in ``config.timebin_origin``.

.. code-block:: python

    data = DetectionData.from_dict({
        "detection_file": Path("APOCADO_results.csv"),
        "filename_format": "%Y-%m-%dT%H-%M-%S",
        "timebin_new": Timedelta("1min"),
    })

Inspecting and adjusting the data
---------------------------------

The ``DataFrame`` is available as ``data.df``, with convenient properties:
``data.annotator``, ``data.label``, ``data.dataset``, ``data.start_datetime`` and ``data.end_datetime``
(a single value is returned as a ``str``, several as a ``list[str]``).

.. note::

    :meth:`~disclose.dataclass.detection_data.DetectionData.reshape` and
    :meth:`~disclose.dataclass.detection_data.DetectionData.change_tz` modify the instance **in place**.
    Use :meth:`~disclose.dataclass.detection_data.DetectionData.copy` to work on an independent instance.

.. code-block:: python

    data.change_tz("Europe/Paris")                    # convert all timestamps
    data.reshape(Timestamp("2022-07-08"), None)       # restrict the period (naive = data timezone)
    data.coordinates = (48.39, -4.49)                 # (lat, lon) of the recorder

    subset = data.filter(annotator="annotator1", label="Buzz")  # new DetectionData
    df = data.filter_df(annotator="annotator1", label="Buzz")   # plain DataFrame

Annotators and labels are paired **element-wise**, not as a cross-product:
``filter(["a1", "a2"], ["l1", "l2"])`` selects ``(a1, l1)`` and ``(a2, l2)``.
A lone ``str`` is broadcast to the length of the other argument.

.. _usage_effort:

Recording effort
----------------

A ``recording_file`` is a CSV with the columns
``start_recording``, ``end_recording``, ``start_deployment`` and ``end_deployment``.
It describes when the recorder was actually recording and can be used to shade periods
without observation effort through ``data.plot.feature(ax, effort=True)``.
See :class:`~disclose.dataclass.recording_effort.RecordingEffort`.

.. code-block:: python

    import matplotlib.pyplot as plt

    data = DetectionData.from_dict({
        "detection_file": Path("ELAN_results.csv"),
        "recording_file": Path("ELAN_recording_periods.csv"),
    })

    fig, ax = plt.subplots()
    data.plot.histogram("annotator1", "Buzz", Timedelta("1D"), ax=ax)
    data.plot.feature(ax, effort=True, bin_size=Timedelta("1D"))

Plotting
--------

Plots are available through ``data.plot`` (see :doc:`plot_accessor`).
Each method accepts an optional keyword-only ``ax``; a new one is created when it is omitted.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Plot
   * - ``histogram``
     - Detections per time bin.
   * - ``polar``
     - Mean proportion of detected bins per hour of the day.
   * - ``heatmap``
     - Hour of day against date (``bin_size`` of at least 24 h).
   * - ``scatter``
     - Detections by hour of day against date.
   * - ``timeline``
     - One row per label.
   * - ``agreement``
     - Inter-annotator agreement of two annotators.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    data.plot.histogram(
        annotator=data.annotator,
        label=data.label[0],
        bin_size=Timedelta("1h"),
        ax=ax,
    )

    polar_ax, _ = data.plot.polar("annotator1", "Buzz")
    data.plot.feature(polar_ax, rise_set=True)

    # Format and overlay a time-based plot independently from its data
    data.plot.set_ax(ax, x_ticks_res=Timedelta("12h"), date_format="%Y-%m-%d")
    data.plot.feature(ax, season=True, rise_set=True, effort=True)

Plots that show sunrise/sunset or the seasons need coordinates:
set ``data.coordinates = (lat, lon)`` before calling ``data.plot.feature``.

Metrics
-------

:meth:`~disclose.dataclass.detection_data.DetectionData.detection_perf` compares two annotator/label pairs.
The **first** pair is the reference.
It returns the precision, recall and F-score, the confusion matrix and the timestamps of each outcome
(see :doc:`api_utils_metric`).

.. code-block:: python

    scores, confusion_matrix, classification = data.detection_perf(
        annotator=("annotator1", "annotator2"),
        label=("Buzz", "Buzz"),
    )
    print(scores.precision, scores.recall, scores.f_score)

The detections must share a single time bin.
If a ``recording_file`` is set, only the recorded bins are taken into account.

Going further
-------------

- :doc:`examples`: notebooks that complete typical tasks.
- :doc:`api`: reference of the classes and utility functions (POD, glider, PAMGuard...).

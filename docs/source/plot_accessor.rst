PlotAccessor
------------

.. _plot_accessor:

``PlotAccessor`` is the plotting namespace of a
:class:`~disclose.dataclass.detection_data.DetectionData`, available as ``data.plot``.
Like in pandas, ``data.plot("polar", ...)`` is equivalent to ``data.plot.polar(...)``.

.. autoclass:: disclose.dataclass.detection_data.PlotAccessor
   :members: histogram, polar, heatmap, scatter, timeline, agreement, set_ax
   :special-members: __call__

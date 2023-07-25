from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt


def text(string: str, format="txt") -> QtWidgets.QLabel:
    """
    creates a label with text
    """
    label = QtWidgets.QLabel()
    label.setText(string)
    if format == "md":
        label.setTextFormat(Qt.MarkdownText)
    return label


class CloseListener(QtCore.QObject):
    """
    detects when a Qt Window has closed. on the closed slot
    for more information please reference https://github.com/napari/napari/issues/4336
    """

    closed = QtCore.Signal()

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self._widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if obj is self.widget and event.type() == QtCore.QEvent.Close:
            self.closed.emit()
        return super().eventFilter(obj, event)

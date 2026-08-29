# Local transport-selection patch

Upstream source: `ros-visualization/rqt_image_view` tag `1.3.0`
(`340d5df45100ab676b8f98caab76393bc91a3719`).

The Taskplanner final overlay is published only through the `compressed`
image transport.  Upstream 1.3.0 stores the transport-aware topic in the
combo box item data (`"/base compressed"`) but selects and persists only the
visible slash label (`"/base/compressed"`).  At startup before graph discovery,
that label is interpreted as a raw `sensor_msgs/Image` topic.

`src/rqt_image_view/image_view.cpp` is patched to select, preserve, and save
the canonical item data.  Legacy slash-form settings remain readable when a
matching discovered entry exists.

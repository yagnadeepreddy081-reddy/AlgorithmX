SAMUDRA NETRA - sample Sentinel-1 tiles
======================================

256 x 256 pixels, 80 m ground sampling, 20.5 km across. This is the tile
size the segmentation network works on; a full Sentinel-1 scene is cut
into these before processing.

Use them with the 'Analyse a new image' button on the console. Four of
these eight contain an oil discharge and four do not. They were generated
from seeds unrelated to the training set and the test set, so the system
has never been run on them.

You can also hand the console any other image at all - it will tell you
whether what it received looks like radar data before it reports anything.

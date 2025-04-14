# globals plot settings
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap

# COLORS
class COLORS:
  SHINY_GREEN = "#bfffbf"
  SHINY_BLUE  = "#a9e7ff"
  MAT_RED     = "#ffc7c7"
  MAT_YELLOW  = "#ffffbd"
  NEON_PINK   = "#ff3ea0"
  BACKGROUND_BLUE = "#1e0c45"
  TEXT_COLOR  = "#ffffff"
  # BHT COLORS
  GREY        ='#555555'
  YELLOW      = '#ffc900'
  ORANGE      = '#ea3b07'
  PETROL      = '#00a0aa'
  BLUE        = '#004282'

HEATMAP = LinearSegmentedColormap.from_list("mycmap", [COLORS.YELLOW,COLORS.ORANGE,COLORS.BLUE])
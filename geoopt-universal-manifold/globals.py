# globals plot settings
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap

# BASE COLORS
class BASE_COLORS:
  SHINY_GREEN = "#bfffbf"
  SHINY_BLUE  = "#a9e7ff"
  MAT_RED     = "#ffc7c7"
  MAT_YELLOW  = "#ffffbd"
  NEON_PINK   = "#ff3ea0"
  BACKGROUND_BLUE = "#1e0c45"
  TEXT_COLOR  = "#ffffff"
  BLACK       = "#000000"
  GREYD       = "#dadad9"
  GREYL       = "#ede6e3"
  # BHT COLORS
  WHITE       = '#ffffff'
  GREY        = '#555555' 
  YELLOW      = '#ffc900'
  ORANGE      = '#ea3b07'
  PETROL      = '#00a0aa'
  BLUE        = '#004282'

# COLOR FUNCTION ASSIGNMENT
class COLORS:
  WHITE     = BASE_COLORS.WHITE
  BOUNDARY  = BASE_COLORS.BACKGROUND_BLUE
  DOMAIN    = BASE_COLORS.GREYL
  GRID  	  = BASE_COLORS.MAT_YELLOW

  VECTOR1   = BASE_COLORS.SHINY_GREEN
  VECTOR2   = BASE_COLORS.PETROL
  LINE      = BASE_COLORS.ORANGE
  POINT     = BASE_COLORS.ORANGE
  VECTOROP  = BASE_COLORS.SHINY_BLUE
  TEXT_COLOR  = "#ffffff"

  GRADIENT    = 'bone'
  RESOLUTION  = 10000
  #LinearSegmentedColormap.from_list("mycmap", [
  # BASE_COLORS.YELLOW,BASE_COLORS.ORANGE,BASE_COLORS.BLUE,BASE_COLORS.BLUE,BASE_COLORS.BLACK], 
  # N=4096)



# GRID SETTINGS

N_GRID_EVALS  = 100
VEC_WIDTH     = 0.02
GRID_LINE_WIDTH = 0.25
FONT_SIZE     = 15
BOUND_WIDTH   = 2.0
# global figure setting for pyplot and mlab
class IMG_DIM:
  def __init__(self,width,square=True,plt=True):
    scale = 1 if plt else 500/7 
    self.WIDTH  = width*scale
    self.HEIGHT = width if square else width*5/7
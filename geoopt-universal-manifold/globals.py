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
  BLACK       = "#000000"
  # BHT COLORS
  WHITE       = '#ffffff'
  GREY        ='#555555'
  YELLOW      = '#ffc900'
  ORANGE      = '#ea3b07'
  PETROL      = '#00a0aa'
  BLUE        = '#004282'

HEATMAP = LinearSegmentedColormap.from_list("mycmap", [
  COLORS.YELLOW,COLORS.ORANGE,COLORS.BLUE,COLORS.BLUE,COLORS.BLACK][::-1], 
  N=4096)


# GRID SETTINGS

N_GRID_EVALS = 1000
# global figure setting for pyplot and mlab
class IMG_DIM:
  def __init__(self,width,square=True,plt=True):
    scale = 1 if plt else 500/7 
    self.WIDTH  = width*scale
    self.HEIGHT = width if square else width*5/7
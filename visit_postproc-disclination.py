# On linux: path/to/visit/bin/visit -nowin -cli -s <script.py>
# On mac: /Applications/VisIt.app/Contents/Resources/bin/visit -nowin -cli -s <script.py>

import argparse
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, default=None)
args, unknown = parser.parse_known_args()

experimentPath = args.input
experimentDir = os.path.dirname(os.path.abspath(experimentPath))
experimentName = os.path.splitext(os.path.basename(experimentPath))[0]

print('Postprocessing {}'.format(experimentPath))

# sys.exit()
OpenDatabase("localhost:"+experimentPath, 0)

steps = TimeSliderGetNStates()

AddPlot("Mesh", "mesh", 1, 1)

MeshAtts = MeshAttributes()
MeshAtts.legendFlag = 1
MeshAtts.lineWidth = 2
MeshAtts.meshColor = (255, 255, 255, 255)
MeshAtts.meshColorSource = MeshAtts.MeshCustom  # Foreground, MeshCustom, MeshRandom
MeshAtts.opaqueColorSource = MeshAtts.Background  # Background, OpaqueCustom, OpaqueRandom
MeshAtts.opaqueMode = MeshAtts.Auto  # Auto, On, Off
MeshAtts.pointSize = 0.05
MeshAtts.opaqueColor = (255, 255, 255, 255)
MeshAtts.smoothingLevel = MeshAtts.Fast  # None, Fast, High
MeshAtts.pointSizeVarEnabled = 0
MeshAtts.pointSizeVar = "default"
MeshAtts.pointType = MeshAtts.Point  # Box, Axis, Icosahedron, Octahedron, Tetrahedron, SphereGeometry, Point, Sphere
MeshAtts.showInternal = 0
MeshAtts.pointSizePixels = 2
MeshAtts.opacity = 0.3
SetPlotOptions(MeshAtts)


AddPlot("Tensor", "Q0", 1, 1)
TensorAtts = TensorAttributes()
TensorAtts.useStride = 1
TensorAtts.stride = 1
# TensorAtts.scale = 
TensorAtts.scaleByMagnitude = 1
TensorAtts.autoScale = 1
TensorAtts.colorByEigenvalues = 1
TensorAtts.useLegend = 1
TensorAtts.tensorColor = (0, 0, 0, 255)
TensorAtts.colorTableName = "Default"
TensorAtts.invertColorTable = 0
SetPlotOptions(TensorAtts)

AddPlot("Pseudocolor", "v", 1, 1)
AddOperator("Displace", 1)
DisplaceAtts = DisplaceAttributes()
DisplaceAtts.factor = 1
DisplaceAtts.variable = "u"
SetOperatorOptions(DisplaceAtts, 0, 1)
AddOperator("Elevate", 1)
ElevateAtts = ElevateAttributes()
ElevateAtts.useXYLimits = ElevateAtts.Always  # Never, Auto, Always
ElevateAtts.limitsMode = ElevateAtts.OriginalData  # OriginalData, CurrentPlot
ElevateAtts.scaling = ElevateAtts.Linear  # Linear, Log, Skew
ElevateAtts.skewFactor = 1
ElevateAtts.zeroFlag = 0
ElevateAtts.variable = "v"
SetOperatorOptions(ElevateAtts, 1, 1)
DrawPlots()

ResetView()
v = GetView3D()
v.RotateAxis(0,-70)
SetView3D(v)

light0 = LightAttributes()
light0.enabledFlag = 1
light0.type = light0.Ambient  # Ambient, Object, Camera
light0.direction = (0, -1, -1)
light0.color = (255, 255, 255, 255)
light0.brightness = 1
SetLight(0, light0)

SaveWindowAtts = SaveWindowAttributes()
SaveWindowAtts.outputToCurrentDirectory = 1
SaveWindowAtts.outputDirectory = experimentDir
# SaveWindowAtts.fileName = "test"
SaveWindowAtts.family = 1
SaveWindowAtts.format = SaveWindowAtts.PNG  # BMP, CURVE, JPEG, OBJ, PNG, POSTSCRIPT, POVRAY, PPM, RGB, STL, TIFF, ULTRA, VTK, PLY
SaveWindowAtts.width = 1024
SaveWindowAtts.height = 1024
SaveWindowAtts.screenCapture = 0
SaveWindowAtts.saveTiled = 0
SaveWindowAtts.quality = 100
SaveWindowAtts.progressive = 0
SaveWindowAtts.binary = 0
SaveWindowAtts.stereo = 0
SaveWindowAtts.compression = SaveWindowAtts.PackBits  # None, PackBits, Jpeg, Deflate
SaveWindowAtts.forceMerge = 0
SaveWindowAtts.resConstraint = SaveWindowAtts.ScreenProportions  # NoConstraint, EqualWidthHeight, ScreenProportions
SaveWindowAtts.advancedMultiWindowSave = 0

RenderingAtts = RenderingAttributes()
RenderingAtts.antialiasing = 1
SetRenderingAttributes(RenderingAtts)

for step in range(steps):
	TimeSliderSetState(step)
	SaveWindowAtts.fileName = "%s"%(os.path.join(experimentDir,experimentName))
	print(SaveWindowAtts.fileName)
	SetSaveWindowAttributes(SaveWindowAtts)

	SaveWindow()

# from visit_utils import encoding

# input_pattern = os.path.join(experimentDir, "{}%04d.png".format(experimentName))
# # %(experimentName, '%04d')
# output_movie = os.path.join(experimentDir, "{}.mpg".format(experimentName))
# encoding.encode(input_pattern,output_movie,fdup=4)




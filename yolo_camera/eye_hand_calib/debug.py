from DHGrasperInterface import DHGrasperInterface as GrasperInterface
try:
    grasper = GrasperInterface("/dev/ttyUSB0")
except:
    print("Failed to initialize EPG Interface")

r = grasper.get_position()
print(r)

grasper.move(500,50,50)
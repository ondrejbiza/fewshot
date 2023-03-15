# Pick and Place via Object and Action Warping

## Start-up

```
cd launch
python main.py
```

Then, make sure all three cameras are running. You can check that in rviz.

Use Ctrl+C to close main.py and wait. If the shutdown doesn't finish properly, some drivers might keep running and you will need to restart the PC.

## Troubleshooting

* One of the cameras repeatedly doesn't start. First, plug and unplug all USBs from the PC and restart main.py. If that doesn't help, restart the computer.
* Execution error. The UR5 driver disconnects from time to time. Restart main.py.
* Planning errors even for simple motions. The reference frame of UR5 sometimes breaks. Restart the robot.
* Protective stop. Enable the robot and restart main.py.
* Robot twists onto itself. On the robot's table, go to program robot -> move -> home. Slowly move the robot to its home position.

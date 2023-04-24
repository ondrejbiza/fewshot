# Pick and Place via Object and Action Warping

## Generating a latent space of object warping

Generate and show, e.g., a latent space of mugs:
```
python -m scripts.learn_warp data/pca_ndf_mugs.pkl ndf_mugs --alpha 0.01 --n-dimensions 8 --pick-canon-warp
python -m scripts.viz.show_latent_space pca_ndf_mugs
```

Generate latent space of mugs, bowls, bottles, trees and boxes:
```
./shell_scripts/learn_warp.sh
```

## Real-world

### Start-up

```
cd launch
python main.py
```

Then, make sure all three cameras are running. You can check that in rviz.

Use Ctrl+C to close main.py and wait. If the shutdown doesn't finish properly, some drivers might keep running and you will need to restart the PC.

### Commands

```python -m scripts.real_world.record_demo bowl_on_mug pick.pkl place.pkl -c```

```python -m scripts.real_world.pick_place bowl_on_mug pick.pkl place.pkl -c```

### Troubleshooting

* One of the cameras repeatedly doesn't start. First, plug and unplug all USBs from the PC and restart main.py. If that doesn't help, restart the computer.
* Execution error. The UR5 driver disconnects from time to time. Restart main.py.
* Planning errors even for simple motions. The reference frame of UR5 sometimes breaks. Restart the robot.
* Protective stop. Enable the robot and restart main.py.
* Robot twists onto itself. On the robot's table, go to program robot -> move -> home. Slowly move the robot to its home position.

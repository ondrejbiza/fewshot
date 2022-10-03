import pybullet as pb


class PybulletObject:

    def __init__(self, object_type_id, object_id):
        self.object_type_id = object_type_id
        self.object_id = object_id

    def get_x_position(self):
        return self.get_position()[0]

    def get_y_position(self):
        return self.get_position()[1]

    def get_xy_position(self):
        return self.get_position()[:2]

    def get_z_position(self):
        return self.get_position()[2]

    def get_position(self):
        pos, _ = pb.getBasePositionAndOrientation(self.object_id)
        return list(pos)

    def get_rotation(self):
        _, rot = pb.getBasePositionAndOrientation(self.object_id)
        return list(rot)

    def get_pose(self):
        pos, rot = pb.getBasePositionAndOrientation(self.object_id)
        return list(pos), list(rot)

    def get_bounding_box(self):
        return list(pb.getAABB(self.object_id))

    def get_contact_points(self):
        return pb.getContactPoints(self.object_id)

    def is_touching(self, obj):
        contact_points = self.get_contact_points()
        for p in contact_points:
            if p[2] == obj.object_id:
                return True
        return False

    def reset_pose(self, pos, rot):
        pb.resetBasePositionAndOrientation(self.object_id, pos, rot)

    def set_color(self, color):

        if len(color) == 3:
            color = [*color, 1.0]
        assert len(color) == 4
        pb.changeVisualShape(self.object_id, -1, rgbaColor=color)

    def __eq__(self, other):
        if not isinstance(other, PybulletObject):
            return False
        return self.object_id == other.object_id and self.object_type_id == other.object_type_id

    def __hash__(self):
        return self.object_id

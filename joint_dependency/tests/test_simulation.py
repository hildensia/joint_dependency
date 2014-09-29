import unittest
import numpy as np

from joint_dependency.simulation import World, Joint, MultiLocker


class TestMultiLocker(unittest.TestCase):
    def setUp(self):
        self.world = World([])

        open_at = np.random.randint(180)
        self.open_d = (max(open_at - 10, 0),
                  max(open_at + 10, 180))
        # The 'handle'
        states = [self.open_d[0], self.open_d[1]]
        dampings = [15, 200, 15]
        self.world.add_joint(Joint(states, dampings, [0, 180], 10e-3))

        # The 'window'
        states = [0]
        dampings = [15, 15]
        self.world.add_joint(Joint(states, dampings, [0, 180], 10e-3))

        self.locker = MultiLocker(self.world, locker=self.world.joints[0],
                                  locked=self.world.joints[1],
                                  locks=[(0, self.open_d[0]),
                                         (self.open_d[1], 180)])


    def test_locker_locks(self):
        self.world.joints[0].q = 0
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)

        self.world.joints[0].q = 0 + self.open_d[0]/2
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)

        self.world.joints[0].q = self.open_d[0]
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)

        self.world.joints[0].q = self.open_d[0] + 1
        self.world.step(.1)
        self.assertFalse(self.world.joints[1].locked)

        self.world.joints[0].q = self.open_d[1] - 1
        self.world.step(.1)
        self.assertFalse(self.world.joints[1].locked)

        self.world.joints[0].q = (self.open_d[1] + self.open_d[0])/2
        self.world.step(.1)
        self.assertFalse(self.world.joints[1].locked)

        self.world.joints[0].q = self.open_d[1] + 1
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)

        self.world.joints[0].q = (self.open_d[1] + 180)/2
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)

        self.world.joints[0].q = 180
        self.world.step(.1)
        self.assertTrue(self.world.joints[1].locked)


class TestJoint(unittest.TestCase):
    def setUp(self):
        self.world = World([])

        states = [0]
        dampings = [15, 15]
        self.world.add_joint(Joint(states, dampings, [0, 180], 10e-3))

    def test_movement(self):
        self.world.joints[0].vel = 10.

        for i in range(10):
            old_vel = self.world.joints[0].vel
            old_q = self.world.joints[0].q
            self.world.step(.1)
            self.assertLess(self.world.joints[0].vel, old_vel)
            self.assertGreater(self.world.joints[0].q, old_q)

    def test_locking(self):
        self.world.joints[0].lock()
        self.assertTrue(self.world.joints[0].is_locked())

        self.world.joints[0].unlock()
        self.assertFalse(self.world.joints[0].is_locked())

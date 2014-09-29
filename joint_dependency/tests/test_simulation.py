import unittest
import numpy as np

from joint_dependency.simulation import (World, Joint, MultiLocker, Furniture,
                                         create_furniture,
                                         create_drawer_with_key)


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


class TestFurniture(unittest.TestCase):
    def setUp(self):
        self.world = World([])

    def test_drawer_with_key(self):
        drawer_min = np.random.randint(0, 100)
        drawer_max = drawer_min + np.random.randint(41, 180-drawer_min)
        handle_min = np.random.randint(0, 100)
        handle_max = handle_min + np.random.randint(41, 180-handle_min)

        create_drawer_with_key(self.world, 10e-3, [(handle_min, handle_max),
                                                   (drawer_min, drawer_max)])

        self.assertEqual(len(self.world.joints), 2)
        self.assertEqual(self.world.joints[0].min_limit, handle_min)
        self.assertEqual(self.world.joints[0].max_limit, handle_max)
        self.assertEqual(self.world.joints[1].min_limit, drawer_min)
        self.assertEqual(self.world.joints[1].max_limit, drawer_max)

        for listener in self.world.listeners:
            if isinstance(listener, MultiLocker):
                self.assertEqual(listener.locks[0][0], handle_min)
                self.assertEqual(listener.locks[1][1], handle_max)
                self.assertLess(listener.locks[0][0], listener.locks[0][1])
                self.assertLess(listener.locks[0][1], listener.locks[1][0])
                self.assertLess(listener.locks[1][0], listener.locks[1][1])
                self.assertLessEqual(listener.locks[1][0]-listener.locks[0][1],
                                     20)
                self.assertGreaterEqual((listener.locks[1][0] -
                                         listener.locks[0][1]), 10)

    def test_create_furniture_wrong_furniture(self):
        self.assertRaises(TypeError, create_furniture, None)

    def test_create_furniture_drawer_with_key(self):
        drawer_min = np.random.randint(0, 100)
        drawer_max = drawer_min + np.random.randint(40, 180-drawer_min)
        key_min = np.random.randint(0, 100)
        key_max = key_min + np.random.randint(40, 180-key_min)

        create_furniture(Furniture.drawer_key, self.world, 10e-3,
                         [(key_min, key_max),
                          (drawer_min, drawer_max)])

        self.assertEqual(len(self.world.joints), 2)
        self.assertEqual(self.world.joints[0].min_limit, key_min)
        self.assertEqual(self.world.joints[0].max_limit, key_max)
        self.assertEqual(self.world.joints[1].min_limit, drawer_min)
        self.assertEqual(self.world.joints[1].max_limit, drawer_max)

        for listener in self.world.listeners:
            if isinstance(listener, MultiLocker):
                self.assertLess(listener.locks[0][0], listener.locks[0][1])
                self.assertLess(listener.locks[0][1], listener.locks[1][0])
                self.assertLess(listener.locks[1][0], listener.locks[1][1])
                self.assertEqual(listener.locks[0][0], key_min)
                self.assertEqual(listener.locks[1][1], key_max)
                self.assertLessEqual(listener.locks[1][0]-listener.locks[0][1],
                                     20)
                self.assertGreaterEqual((listener.locks[1][0] -
                                         listener.locks[0][1]), 10)

    def test_create_furniture_drawer_with_handle(self):
        drawer_min = np.random.randint(0, 100)
        drawer_max = drawer_min + np.random.randint(20, 180-drawer_min)
        handle_min = np.random.randint(0, 100)
        handle_max = handle_min + np.random.randint(20, 180-handle_min)

        create_furniture(Furniture.drawer_handle, self.world, 10e-3,
                         [(handle_min, handle_max),
                          (drawer_min, drawer_max)])

        self.assertEqual(len(self.world.joints), 2)
        self.assertEqual(self.world.joints[0].min_limit, handle_min)
        self.assertEqual(self.world.joints[0].max_limit, handle_max)
        self.assertEqual(self.world.joints[1].min_limit, drawer_min)
        self.assertEqual(self.world.joints[1].max_limit, drawer_max)

        for listener in self.world.listeners:
            if isinstance(listener, MultiLocker):
                self.assertLess(listener.locks[0][0], listener.locks[0][1])

    def test_create_furniture_cupboard_with_key(self):
        cupboard_min = np.random.randint(0, 100)
        cupboard_max = cupboard_min + np.random.randint(40, 180-cupboard_min)
        key_min = np.random.randint(0, 100)
        key_max = key_min + np.random.randint(40, 180-key_min)

        create_furniture(Furniture.cupboard_key, self.world, 10e-3,
                         [(key_min, key_max),
                          (cupboard_min, cupboard_max)])

        self.assertEqual(len(self.world.joints), 2)
        self.assertEqual(self.world.joints[0].min_limit, key_min)
        self.assertEqual(self.world.joints[0].max_limit, key_max)
        self.assertEqual(self.world.joints[1].min_limit, cupboard_min)
        self.assertEqual(self.world.joints[1].max_limit, cupboard_max)

        for listener in self.world.listeners:
            if isinstance(listener, MultiLocker):
                self.assertLess(listener.locks[0][0], listener.locks[0][1])
                self.assertLess(listener.locks[0][1], listener.locks[1][0])
                self.assertLess(listener.locks[1][0], listener.locks[1][1])
                self.assertEqual(listener.locks[0][0], key_min)
                self.assertEqual(listener.locks[1][1], key_max)
                self.assertLessEqual(listener.locks[1][0]-listener.locks[0][1],
                                     20)
                self.assertGreaterEqual((listener.locks[1][0] -
                                         listener.locks[0][1]), 10)

    def test_create_furniture_cupboard_with_handle(self):
        drawer_min = np.random.randint(0, 100)
        drawer_max = drawer_min + np.random.randint(20, 180-drawer_min)
        handle_min = np.random.randint(0, 100)
        handle_max = handle_min + np.random.randint(20, 180-handle_min)

        create_furniture(Furniture.drawer_handle, self.world, 10e-3,
                         [(handle_min, handle_max),
                          (drawer_min, drawer_max)])

        self.assertEqual(len(self.world.joints), 2)
        self.assertEqual(self.world.joints[0].min_limit, handle_min)
        self.assertEqual(self.world.joints[0].max_limit, handle_max)
        self.assertEqual(self.world.joints[1].min_limit, drawer_min)
        self.assertEqual(self.world.joints[1].max_limit, drawer_max)

        for listener in self.world.listeners:
            if isinstance(listener, MultiLocker):
                self.assertLess(listener.locks[0][0], listener.locks[0][1])

import numpy as np
from physics_sim import PhysicsSim

class Task2():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Original reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = 1.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = np.tanh(-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum()) # Normalizing the values between -1:1
        
        # Experiments with other reward functions:
        #
        # For TAKEOFF, the main component that I have to evaluate is z. 
        # We want to keep an eye on x and y while we are taking off as we don't want to crash (with any object).
        # However, we should emphasize checking z as we want the quadcopter to go up.
        # We could reward positively if quadcopter is going up. An negatively if it goes down.
        # 
        # diff_pos = self.sim.pose[:3] - self.target_pos 
        # reward = 1.0-0.2*abs(diff_pos[0]+diff_pos[1]).sum()-5*np.tanh(diff_pos[2])
        # reward = 1.0-0.2*np.tanh(diff_pos[0])-0.2*np.tanh(diff_pos[1])-1.0*np.tanh(diff_pos[2])
        # reward = 1.0-0.02*(abs(diff_pos[0]+diff_pos[1])).sum()-0.10*np.tanh(diff_pos[2])
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
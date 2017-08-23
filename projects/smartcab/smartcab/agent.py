import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

"""
# Notes From Reviewer

I am guessing these charts are with all 5 features. I think the issue is that you are decaying your epsilon too quickly.
You need to spend more time exploring so you agent can learn the optimal actions before testing. You can also increase
the number of testing trials above 10 (the default) as this will help reduce any bad luck tests. But one major accident
will ruin your safety rating.

Here are some general advices about the parameters.

# epsilon manages exploration vs exploitation

The epsilon system is designed to manage the exploration vs exploitation system.

It's a constant battle in reinforcement learning to balance exploration with exploitation. That is trying new actions
in order to find the optimal policy, while exploiting enough to be successful at the task.

In this simulation, it's actually possible to just generate enough trials in training that we can entirely learn the
optimal policy. However more complicated scenarios may have much large state spaces in which the agent will never
explore them all, so we need to more carefully balance exploration with exploitation. That's the goal of epsilon decay,
and there is a lot of active research in this area.

# epsilon tolerance

epsilon tolerance is a value that triggers the switch from building our agent up with exploration/exploitation aka
training (updating our q-tables) to testing, which just evaluates the agent we have. Setting a lower tolerance will
increase the amount of training trials (updating the q-table), before testing (evaluating our agent) begins. Setting
tolerance high will make the agent decrease training times.

Keep in mind that if your epsilon decays very slowly, then you will have more training trials as well, and if your
epsilon decays very quickly, you will have less training trials as well. So in some sense, the epsilon decay and
epsilon tolerance both control the same "knob", which is "how long do we train for?"

alpha

Alpha is the learning rate. It's saying "how much should my agent 'remember' from this experience"
If we look into the equation, which generally looks like this
self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * reward
or
self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
then we see that alpha intuitively updates the reward, the current q-value, and puts them together.

If we have a small alpha we basically only update a little bit of the reward and mostly keep what we already knew.
if we have a large alpha, we update largely based on the reward.

However -- we need to keep in mind the following :

Our environment is deterministic:

Given any state, the agent will always get the same exact reward for the same exact action (this is actually not
completely true, there is a random nature to reward assignment, but the "good" rewards are always a step higher than
the "bad" rewards). This deterministic property of reward assignment, makes alpha arbitrary (as long as it's >0). No
matter how the rewards are updated (with large or small alpha's), the agent will get the largest value for "good"
actions and smaller values for bad actions. So it should learn the optimal policy either way.

Keep in mind, this is an edge case of q-learning - it applies because our Q-learning has Gamma = 0, there are no
future rewards to consider (and balance against the alpha parameter) and also because our environment is deterministic
in how rewards are assigned.
"""

class LearningAgent(Agent):
    """
    An agent that learns to drive in the Smartcab world.
    This is the object you will be modifying.

    """
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.t = 1

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
            # self.epsilon = self.epsilon - 0.05

            # Decays too fast.
            # self.epsilon = 0.50**self.t

            # Exponential decay
            alpha=0.001
            self.epsilon = math.exp(-alpha*self.t)

            # if self.epsilon > 0.0:
            #     self.epsilon_decay = 1.0 / 200000 * self.t
            #     self.epsilon = self.epsilon - self.epsilon_decay

            # self.epsilon = 1/(self.t**2)

            self.t += 1

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment

        # Which is the direction the Smartcab should drive leading to the destination, relative to the
        # Smartcab's heading.
        waypoint = self.planner.next_waypoint() # The next waypoint

        # Which is the sensor data from the SmartCab including:
        #   'light': the color of the light
        #   'left': the intended direction of travel for a vehicle to the SmartCab's left.
        #           Return None if no vehicle is present.
        #   'right': the intended direction of travel for a vehicle to the SmartCab's left.
        #            Return None if no vehicle is present.
        #   'oncoming': the intended direction of travel for a vehicle across the intersections from the SmartCab.
        #               Return None if no vehicle is present.
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic

        #   'deadline': which is the number of actions remaining for the SmartCab to reach the destination
        #               before running out of time.
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'])
        # state = (waypoint, inputs['light'], inputs['oncoming'])

        return state

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        return max(self.Q[state].items(), key=lambda x: x[1])[1]

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        if state not in self.Q:
            self.Q[state] = {}

            for key in self.valid_actions:
                self.Q[state][key] = 0.0

        return self.Q

    def choose_action(self, state):
        """
        The choose_action function is called when the agent is asked to choose
        which action to take, based on the 'state' the smartcab is in.
        :params state:
        """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state

        # Notes
        # exploit: the knowledge that it has found for the current state s by doing one of the actions a that maximizes
        #           Q[s, a].
        # explore: in order to build a better estimate of the optimal Q-function.  That is, it should select a \
        #           different action from the one that it currently thinks is best.

        if self.learning:
            max_Q = self.get_maxQ(state)

            if self.epsilon > random.random():
                # From time to time, choose a random action.
                action = random.choice(self.valid_actions)
            else:
                # Choose Max Q
                valid_actions = []

                for key in self.valid_actions:
                    if self.Q[state][key] == max_Q:
                        valid_actions.append(key)

                action = random.choice(valid_actions)

        else:
            # Choose Random State
            action = random.choice(self.valid_actions)

        return action

    def learn(self, state, action, reward):
        """
        The learn function is called after the agent completes an action and
        receives an award. This function does not consider future rewards
        when conducting learning.

        """

        ###########
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward - self.Q[state][action])

        return

    def update(self):
        """
        The update function is called when a time step is completed in the
        environment for a given trial. This function will build the agent
        state, choose an action, receive a reward, and learn if enabled.
        """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    state = 'optimized'

    learning = False
    epsilon = 1
    alpha = 0.5
    update_delay = 0.01
    display = False
    log_metrics = True
    optimized = False
    tolerance = 0.05
    n_test = 0

    if state == 'unoptimized':
        update_delay = 0.01
        n_test = 10
        learning = True

    elif state == 'optimized':
        learning = True
        n_test = 1000

        update_delay=0.005
        epsilon=1.0
        alpha=0.5
        log_metrics=True
        optimized=True
        tolerance=0.001

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=False)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(
        LearningAgent,
        learning=learning,
        epsilon=epsilon,
        alpha=alpha
    )
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(
        agent,
        enforce_deadline=True
    )

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(
        env,
        update_delay=update_delay,
        display=display,
        log_metrics=log_metrics,
        optimized=optimized
    )
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(
        tolerance=tolerance,
        n_test=n_test
    )


if __name__ == '__main__':
    run()

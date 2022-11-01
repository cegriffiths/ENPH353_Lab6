import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        # Load stored values
        with open(filename) as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        # Save stored values
        print("Storing: {}".format(self.q))
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # Find the rewards associated with each action in the current state
        # self.newState(state)
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        minQ = min(q)
        rand = False

        # If random value > epsilon then at add a random amount proportional to the 
        # magnitude of the rewards for the rewards for this decision
        if random.random() < self.epsilon:


            # mag = max(abs(minQ), abs(maxQ))
            # recalculate maxQ after adding random values to all actions
            # q = [q[i] + random.random() * mag - 0.5 * mag
            #      for i in range(len(self.actions))]
            # q = [q[i] + 2 * random.random() * mag - mag
            #      for i in range(len(self.actions))]
            # maxQ = max(q)
            # q = [maxQ for a in self.actions]
            rand = True

            i = random.randint(0,2)
        else:
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

        action = self.actions[i]

        # print("State: " + str(state) + "\nLeft: " + str(self.getQ(state, 1)) + 
        #     "\tForward: " + str(self.getQ(state, 0)) + "\tRight " + str(self.getQ(state, 2)) + 
        #     "\nRand: " + str(rand) + "\tAction: " + str(action))

        print("State: %s\nLeft: %0.2f\tForward: %0.2f\tRight: %0.2f\nRand: %d\tAction: %d" % 
            (state, self.getQ(state, 1), self.getQ(state, 0), self.getQ(state, 2), rand, action))

        if return_q:
            return action, q
        return action
        
        
        # THE NEXT LINE NEEDS TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        #return self.actions[1]

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma * max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # print("Q: " + str(self.q) + "\tState1: " + str(state1) + "\taction1: " + str(action1) + 
        #     "\nreward" + str(reward) + "\tstate2" + str(state2))

        maxqnew = max([self.getQ(state2, a) for a in self.actions])

        oldv = self.q.get((state1, action1), None)
        # newReward = oldv + self.alpha * (reward + 
        #         self.gamma * maxqnew - oldv)

        if oldv is None:
            self.q[(state1, action1)] = reward
        elif reward < oldv + self.alpha * (reward + 
                self.gamma * maxqnew - oldv):
            self.q[(state1, action1)] = oldv + self.alpha * (reward + 
                self.gamma * maxqnew - oldv)
            
        # self.num_times_learn += 1

        # THE NEXT LINE NEEDS TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        # self.q[(state1,action1)] = reward

    def newState(self, state):

        alreadyContains = False
        for a in self.actions:
            if (state, a) in self.q.keys() and self.getQ(state, a) != 0:
                alreadyContains = True
        
        if not(alreadyContains):
            print("\nCreating new state!\n\n")
            for ac in self.actions:
                self.q[(state, ac)] = 0.0
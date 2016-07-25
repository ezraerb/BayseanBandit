# Simple baysean bandit. A bandit problem involes a series of choices which
# produce rewards with various probabilities. The goal is to make choices in a 
# way that maximizes the total possible reward. An effective algorithm needs to 
# balance between trying choices that may be better than the current best choice
# (exploration) and trying the best choice found so far (exploitation). Baysean
# bandit algorithms keep a calculated distribution of likely rewards for each 
# choice, sample those distributions to find the best choice each round, and 
# update the distribution as results are obtained. They are one of the most 
# efficient algorithms, especially for the common case of the Bernoulli Bandit 
# (either a reward appears or it doesn't, with constant probablity per bandit)
#
# Most code to learn bandit algorithms is done on simulated data, because 
# real data is hard to obtain in practice. This code follows that lead
#
#   Copyright (C) 2016   Ezra Erb
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#   I'd appreciate a note if you find this program useful or make
#   updates. Please contact me through LinkedIn or github (my profile also has
#   a link to the code depository)
#
import sys
import numpy as np
import matplotlib.pyplot as plt

# A set of bandits. Initialize with the count. 
class Bandits():
      def __init__(self, size):
          '''
          Initialize set of bandits with the number. Must have at least two for
          a valid setup. These are Bernouli bandits, so each gets a probability
          of success when selected
          '''
          if size < 2: # Must have at least two for a valid test
             raise ValueError('Number of bandits for test is invalid, must specify at least two')
          self.bandits = np.random.random_sample(size)
          # Store the maximum probability, used for regret calculation
          self.max_prob = np.amax(self.bandits)

      def select(self, number):
          ''' Select a bandit, retuns whether it poduces an award or not '''
          if (number < 0) or (number >= self.bandits.size):
             return False # Out of range
          return np.random.binomial(1, self.bandits[number]) == 1

      # WARNING: This method can easily be used to cheat and find the best one. 
      # The caller is responsible for avoiding this!
      def regret(self, number):
          ''' 
          Returns the regret of the selected bandit. This is defined as how
          much reward the caller lost by selecting this bandit instead of the
          best one
          '''
          if (number < 0) or (number > self.bandits.size):
             return self.max_prob # out of range
          else:
             return self.max_prob - self.bandits[number]

      def __repr__(self):
          return 'Bandits: ' + self.bandits.__str__()

      def __str__(self):
          return self.__repr__()

def draw_bandit_distribution(stats):
    ''' 
    For each bandit, calculate the probability that the bandit will produce a
    reward given its rewards and losses so far. Calculated using the Beta
    Bernouli distribution
    '''

    # Passing a structured type as an argument looses the field names, requring
    # the following
    return np.random.beta(stats[0] + 1, stats[1] + 1)

draw_bandit_distribution = np.vectorize(draw_bandit_distribution, otypes=[np.float])

if len (sys.argv) != 3:
   print "Usage [number of bandits] [number of attempts]"
else:
   bandit_count = int(sys.argv[1])
   trial_count = int(sys.argv[2])
   if trial_count < 1:
      trial_count = 1

   bandits = Bandits(bandit_count)
   # Print the bandits to make verification of results easier
   print bandits

   banditStats = np.zeros((bandit_count,), dtype=[('wins', np.int ), ('losses', np.int)])
   overallStats = np.zeros((trial_count,), dtype=[('bandit', np.int), ('wins', np.int ), ('losses', np.int), ('regret', np.float)])

   for iteration in range(trial_count):
      # copy statistics
      if (iteration != 0):
         overallStats[iteration] = overallStats[iteration - 1].copy()

      # Draw from the existing model distribution from each bandit.
      current_draws = draw_bandit_distribution(banditStats)

      # Find the one with the highest value and select it
      selected_bandit = current_draws.argmax(axis=0)
      overallStats[iteration]['bandit'] = selected_bandit
      if bandits.select(selected_bandit):
         # Reward!
         banditStats[selected_bandit]['wins'] += 1
         overallStats[iteration]['wins'] += 1
      else:
         # Failed
         banditStats[selected_bandit]['losses'] += 1
         overallStats[iteration]['losses'] += 1
      overallStats[iteration]['regret'] += bandits.regret(selected_bandit)

   # Scatterplot of chosen bandits, followed by plot of total regret
   plt.figure(1)
   plt.subplot(211)
   plt.scatter(np.arange(trial_count), overallStats['bandit'])
   plt.ylabel('chosen bandit')
   plt.xlabel('attempts')

   plt.subplot(212)
   plt.plot(overallStats['regret'])
   plt.ylabel('total regret')
   plt.xlabel('attempts')
   plt.show()
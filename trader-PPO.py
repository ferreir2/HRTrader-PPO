""" Implements a Proximal Policy Optimization (PPO) RL algorithm
    for horse racing online trading.
    Based on J. Schulman, et al, arXiv:1707.06347v2 [cs.LG] 2017
    and OpenAI PPO implementation: 
    http://spinningup.openai.com/en/latest/algorithms/ppo.html
 """

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pickle
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------------------------
# 
#  HELPER FUNCTIONS
# 
# --------------------------------------------------------------------------------------------------------------


def size_lay_to_close_back_bet(price_back, size_back, price_lay, gamma=0.95):
    """ Returns the size of a lay bet to close a back bet at the current price [price_lay]
        gamma is the profit discount from betfair commission """
    return size_back * (1 + gamma*(price_back - 1)) / (gamma + price_lay - 1)


def size_back_to_close_lay_bet(price_lay, size_lay, price_back, gamma=0.95):
    """ Returns the size of a back bet to close a lay bet at the current price [price_back]
        gamma is the profit discount from betfair commission """
    return size_lay * (gamma + price_lay - 1) / (1 + gamma*(price_back - 1))


def get_pnl(side, price, size, gamma=0.95):
    """ Returns the pnl for a [side] bet of £[size] at price [price] """
    if side == 'back':
        pnl_wins = gamma * size * (price - 1)
        pnl_looses = -size
    elif side == 'lay':
        pnl_wins = - size * (price - 1)
        pnl_looses = gamma * size
    else:
        raise ValueError(f"Invalid side: {side}")
    
    return pnl_wins, pnl_looses


def action_label(act, obs):
    """ Returns the action label acording to the current state.
        As a convention, the last 3 entries of the state (observation) encode the current book state.
    """
    if obs[-1] == 1: # back bet open
        act_dict = {0: 'hold', 1: 'close_back_bet', 2: 'back'}
    elif obs[-2] == 1: # lay bet open
        act_dict = {0: 'hold', 1: 'lay', 2: 'close_lay_bet'}
    elif obs[-3] == 1: # no bet open
        act_dict = {0: 'hold', 1: 'lay', 2: 'back'}
    else:
        raise ValueError(f'bad state: {obs}')
    
    return act_dict[act]


def build_policy_mlp(obs_ph, layer_sizes, act_dim, vscope='pi'):
    """ Builds the policy mlp """
    
    with tf.variable_scope(vscope):
        pi_logits = tf.layers.dense(obs_ph, units=layer_sizes[0], activation=tf.nn.relu)
        for size in layer_sizes[1:]:
            pi_logits = tf.layers.dense(pi_logits, units=size, activation=tf.nn.relu)
        pi_logits = tf.layers.dense(pi_logits, units=act_dim)
    
        log_pi = tf.nn.log_softmax(pi_logits)
        pi = tf.exp(log_pi)

    return pi_logits, log_pi, pi


def build_value_function_mlp(obs_ph, layer_sizes, vscope='vf'): 
    """ Builds the value function mlp """
    
    with tf.variable_scope(vscope):
        val_func = tf.layers.dense(obs_ph, units=layer_sizes[0], activation=tf.nn.relu)
        for size in layer_sizes[1:]:
            val_func = tf.layers.dense(val_func, units=size, activation=tf.nn.relu)
        val_func = tf.layers.dense(val_func, units=1)

    return val_func


def kld(new_logps, old_logps):
    """ KL divergence """
    return tf.reduce_mean(old_logps - new_logps)


def plot_trade(probs, act_labels, price_back, price_lay):
    """ Plots the actions and corresponding probabilities for a single runners. """

    _, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    axs[0].plot(price_back, color='b')
    axs[0].plot(price_lay, color='r')

    i_back_bet = np.where(act_labels == 'back')[0]
    i_lay_bet = np.where(act_labels == 'lay')[0]
    i_close_back_bet = np.where(act_labels == 'close_back_bet')[0]
    i_close_lay_bet = np.where(act_labels == 'close_lay_bet')[0]

    _ = axs[0].plot(i_back_bet, price_back[i_back_bet], linestyle='none', marker='v')
    _ = axs[0].plot(i_lay_bet, price_lay[i_lay_bet], linestyle='none', marker='^')
    _ = axs[0].plot(i_close_back_bet, price_lay[i_close_back_bet], linestyle='none', marker='x', c='k')
    _ = axs[0].plot(i_close_lay_bet, price_back[i_close_lay_bet], linestyle='none', marker='x', c='k')

    _ = axs[1].plot(probs)
    plt.legend(['Hold','Lay/Close','Back/Close'])


# --------------------------------------------------------------------------------------------------------------
# 
#  ENVIRONMENT
# 
# --------------------------------------------------------------------------------------------------------------


class Env(object):
    """ Implements the environment seen by the agent. Methods are inspired in the openAI design.
        By design, all pre-processing required on the market data (prices, volumes, etc) is performed on the environment side.
        The state returned by the methods 'step' and 'reset' should be ready for the agent's mlp.
    """

    def __init__(self, path, bet_size=1):
        
        # load markets from file with pre-processed races to save time during training.
        # Assumes a pickled list of Pandas dataframes
        with open(path, mode='rb') as f:
            self.samples = pickle.load(f)

        # Define internal variables
        self.done = False # True if episode is over
        self.n_runners = len(self.samples) # Total number of runners in sample
        self.runner_index = None # Runner index (dependent on pre-processing)
        self.ptr = None # Pointer (int) to current position
        self.end_ptr = None # Pointer (int) to signal the end of the episode
        self.runner = None # Runner DataFrame
        self.pnls = [] # List to store PnLs during race
        self.book = None # Bet book to keep track of open and closed bets
        self.side = 0 # 0 = no open bet | 1 = lay bet open | 2 = back bet open
        self.bet_size = bet_size # The bet size used by default
        self.gamma = 0.95 # A discount on profits comming from the broker commission
        self.curr_state = None # The current env. state

    def current_state(self):
        """ Defines the state observed by the agent. Implement according to pre-processed races.
            Returns the current state.
        """ 

        return self.curr_state

    def reset(self, runner_index=None, ptr_index=None):
        """ Reset the environment. Implement according to pre-processed races.
            Returns the current state"""
        
        return self.curr_state
    
    def step(self, action):
        """ Act on environment according to action. The action taken (lay, back, close or do nothing)
        depends on the current state.
        """

        # current prices
        price_back = self.runner.loc[self.ptr, 'price_back']
        price_lay = self.runner.loc[self.ptr, 'price_lay']

        if action_label(action, self.curr_state) == 'lay':
            open_side = 'lay'
            open_price = price_lay
            open_size = self.bet_size
            self.add_bet(open_side, open_price, open_size)
            self.side = 1
        
        elif action_label(action, self.curr_state) == 'back':
            open_side = 'back'
            open_price = price_back
            open_size = self.bet_size
            self.add_bet(open_side, open_price, open_size)
            self.side = 2
        
        elif 'close' in action_label(action, self.curr_state):
            self.balance_book(price_back, price_lay)
            self.side = 0

        # update pointer and prices
        self.ptr += 1
        price_back = self.runner.loc[self.ptr, 'price_back']
        price_lay = self.runner.loc[self.ptr, 'price_lay']
        
        
        if self.ptr == self.end_ptr: # Close
            self.done = True
            self.balance_book(price_back, price_lay)
        else:
            self.done = False
        
        # Update pnls
        self.pnls.append(self.get_upnl(price_back, price_lay))
        
        # Reward as change in pnl
        rew = self.pnls[-1] - self.pnls[-2]

        # Update current state
        self.curr_state = self.current_state()

        # Env info
        info = {}

        return self.curr_state, rew, self.done, info

    def add_bet(self, open_side, open_price, open_size):
        """ Adds a <open_side> bet of <open_size> at <open_price> """
        self.book = self.book.append({'open_side': open_side, 'open_price': open_price,
                                      'open_size': open_size, 'status': 'open'}, ignore_index=True)

    def balance_book(self, price_back, price_lay):
        """ Closes all open bets. """
        for _, bet in self.book.iterrows():
            if bet.status == 'closed': continue
            
            if bet.open_side == 'lay':
                bet['close_price'] = price_back
                bet['close_size'] = size_back_to_close_lay_bet(bet.open_price, bet.open_size, bet.close_price)
                bet['pnl'] = self.gamma * bet.open_size - bet.close_size
            else:
                bet['close_price'] = price_lay
                bet['close_size'] = size_lay_to_close_back_bet(bet.open_price, bet.open_size, bet.close_price)
                bet['pnl'] = self.gamma * bet.close_size - bet.open_size
            
            bet['status'] = 'closed'

    def get_upnl(self, price_back, price_lay):
        """ Returns the unrealized PnL = PnL from closed bets + PnL from open bets if they were closed at current price"""
        # pnl from closed bets
        pnl = self.book.pnl.sum()

        # upnl from open bets
        for _, bet in self.book.iterrows():
            if bet.status == 'closed': continue
            
            if bet.open_side == 'lay':
                close_price = price_back
                close_size = size_back_to_close_lay_bet(bet.open_price, bet.open_size, close_price)
                pnl += self.gamma * bet.open_size - close_size
            else:
                close_price = price_lay
                close_size = size_lay_to_close_back_bet(bet.open_price, bet.open_size, close_price)
                pnl += self.gamma * close_size - bet.open_size
        
        return pnl


# --------------------------------------------------------------------------------------------------------------
# 
#  AGENT
# 
# --------------------------------------------------------------------------------------------------------------

class Agent(object):
    def __init__(self, pi_mlp_ls, vf_mlp_ls, GAE_gamma=0.99, GAE_lbda=0.97, pi_lr=0.01, vf_lr=0.01,
                 pi_max_iter_per_batch=100, vf_max_iter_per_batch=1000, vf_earlystop=10, pi_clip=0.2):

        # ---- Parameters
        self.GAE_gamma = GAE_gamma
        self.GAE_lbda = GAE_lbda
        self.obs_dim = 16 # Update according to Env.current_state implementation
        self.act_dim = 3
        self.pi_max_iter_per_batch = pi_max_iter_per_batch
        self.vf_max_iter_per_batch = vf_max_iter_per_batch
        self.vf_earlystop = vf_earlystop

        # ---- Placeholders
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim))
        self.act_ph = tf.placeholder(tf.int32, (None, ))
        self.log_pi_act_ph = tf.placeholder(tf.float32, (None, ))
        self.log_pi_ph = tf.placeholder(tf.float32, (None, self.act_dim))
        self.adv_ph = tf.placeholder(tf.float32, (None, ))
        self.r2g_ph = tf.placeholder(tf.float32, (None, ))

        # ---- placeholder labels
        self.ph_labels = {self.obs_ph: 'obs', self.act_ph: 'act', self.log_pi_ph: 'log_pi',
                          self.log_pi_act_ph: 'log_pi_act', self.adv_ph: 'adv', self.r2g_ph: 'r2g'}

        # ---- Policy (pi) mlp and loss function
        self.pi_logits, self.log_pi, self.pi = build_policy_mlp(self.obs_ph, pi_mlp_ls, self.act_dim)
        self.action_mask = tf.one_hot(self.act_ph, self.act_dim)
        self.log_pi_act = tf.reduce_sum(self.action_mask * self.log_pi, axis=1)
        ratio = tf.exp(self.log_pi_act - self.log_pi_act_ph)
        clip = tf.minimum(tf.maximum(ratio, tf.constant(1 - pi_clip)), tf.constant(1 + pi_clip))
        self.pi_loss = - tf.reduce_mean(tf.minimum(ratio * self.adv_ph, clip * self.adv_ph), axis=0)
        self.pi_vars = [v for v in tf.trainable_variables() if 'pi' in v.name]

        # ---- Value function V_{phi_k}(s_t) mlp and loss function
        self.value_func = build_value_function_mlp(self.obs_ph, vf_mlp_ls)
        self.value_func_loss = tf.losses.mean_squared_error(labels=self.r2g_ph, predictions=tf.squeeze(self.value_func))
        self.val_func_vars = [v for v in tf.trainable_variables() if 'vf' in v.name]

        # ---- Train operations
        self.pi_train_step = tf.Variable(0, trainable=False)
        self.pi_lr = tf.train.exponential_decay(pi_lr, self.pi_train_step, 10, 0.96, staircase=False)
        self.pi_train = tf.train.AdamOptimizer(self.pi_lr).minimize(self.pi_loss)
        self.value_func_train = tf.train.AdamOptimizer(vf_lr).minimize(self.value_func_loss)

        # ---- Sample action op
        self.sample_action = tf.squeeze(tf.multinomial(logits=self.pi_logits, num_samples=1), axis=1)

    def update_params(self, sess, variables, params):
        """ Updates tensor values."""
        sess.run(tf.group(*[tf.assign(v, p) for v, p in zip(variables, params)]))
    
    def update_val_func(self, sess, batch_inputs):
        """ Fits value function with early stopping. """

        best_loss, best_params = sess.run([self.value_func_loss, self.val_func_vars], feed_dict=batch_inputs)
        patience = self.vf_earlystop

        # Fit
        for i in range(self.vf_max_iter_per_batch):
            _ = sess.run(self.value_func_train, feed_dict=batch_inputs)
            loss_val, val_func_params = sess.run([self.value_func_loss, self.val_func_vars], feed_dict=batch_inputs)

            if loss_val < best_loss:
                best_loss = loss_val
                best_params = val_func_params
                patience = self.vf_earlystop
            else:
                patience -= 1
            
            if patience == 0: break

        print(f'Number of value function train iteration:', i+1)

        # Update params to best iteration
        self.update_params(sess, self.val_func_vars, best_params)

    def generate_trades(self, sess, env, batch_size, runner_index=None):
        """ Generate trades according to current policy. """

        # Batch lists
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_r2g = []
        batch_adv = []
        batch_pnls = []
        batch_lens = []
        batch_runners_idx = []

        # reset episode-specific variables
        obs = env.reset(runner_index=runner_index)
        done = False
        ep_rews = []
        ep_obs = []

        while True:
            
            # Sample action from pi
            act = sess.run(self.sample_action, {self.obs_ph: np.array([obs])})[0]

            # Store observation and action
            batch_obs.append(obs)
            ep_obs.append(obs)
            batch_acts.append(act)

            # Take action
            obs, rew, done, _ = env.step(act)

            # Store reward
            batch_rews.append(rew)
            ep_rews.append(rew)

            # End of episode
            if done:
                # Store runner index
                batch_runners_idx.append(env.runner_index)

                # Calculate total reward and episode duration
                ep_rew = sum(ep_rews)
                ep_len = len(ep_rews)
                
                # Store PnL and lengths
                batch_pnls.append(ep_rew)
                batch_lens.append(ep_len)

                # Calculate rewards-to-go and store
                r2g = [sum(ep_rews[i:]) for i in range(ep_len)]
                batch_r2g += r2g

                # Calculate advantage function using GAE
                # as described in J. Schulman, et al. (2016) (https://arxiv.org/pdf/1506.02438.pdf)
                ep_val_func = sess.run(tf.squeeze(self.value_func), {self.obs_ph: np.array(ep_obs)})
                
                if not np.isscalar(ep_val_func):
                    deltas = np.array(ep_rews) + np.append(self.GAE_gamma * ep_val_func[1:] - ep_val_func[:-1],  - ep_val_func[-1])
                else:
                    deltas = np.array([ep_rews[0] - ep_val_func])
                
                discounts = (self.GAE_lbda*self.GAE_gamma)**np.arange(ep_len)
                adv = [sum(discounts[:(ep_len - t)] * deltas[t:]) for t in range(ep_len)]
                
                # Store advantages
                batch_adv += adv

                # If it has collected enough observations, break
                if len(batch_obs) > batch_size:
                    break

                # Reset epsiode-specific vars
                obs, done, ep_rews, ep_obs, _ = env.reset(runner_index=runner_index), False, [], [], 0
        
        # Convert lists to arrays
        batch_acts = np.array(batch_acts)
        batch_adv = np.array(batch_adv)
        batch_r2g = np.array(batch_r2g)
        batch_obs = np.array(batch_obs)

        # Get log_pi and log_pi_act
        batch_log_pi = sess.run(self.log_pi, {self.obs_ph: batch_obs})
        batch_log_pi_act = sess.run(tf.reduce_sum(self.action_mask * self.log_pi_ph, axis=1),
                                         {self.act_ph: batch_acts, self.log_pi_ph: np.array(batch_log_pi)})

        batch_inputs = {self.obs_ph: batch_obs, self.act_ph: batch_acts, self.r2g_ph: batch_r2g, self.adv_ph: batch_adv,
                        self.log_pi_ph: batch_log_pi, self.log_pi_act_ph: batch_log_pi_act}

        return batch_inputs, batch_pnls, batch_lens, batch_runners_idx

    def train_on_batch(self, sess, env, batch_size, runner_index=None):
        """ Trains pi and value function on one batch of trades. """

        # Generate trades
        batch_inputs, batch_pnls, batch_lens, batch_runnners_idx = self.generate_trades(sess, env, batch_size, runner_index=runner_index)
        
        # Train network
        pi_params = sess.run(self.pi_vars, batch_inputs)
        kld_value = 0
        current_train_step, curr_lr = sess.run([self.pi_train_step, self.pi_lr])
        
        # Cycle learning rate
        if curr_lr < 0.001:
            current_train_step = 0
            self.update_params(sess, [self.pi_train_step], [current_train_step])
        
        # Train
        for i in range(self.pi_max_iter_per_batch):
            _ = sess.run(self.pi_train, batch_inputs)
            new_kld_value, new_pi_params = sess.run([kld(self.log_pi_act, self.log_pi_act_ph), self.pi_vars], batch_inputs)
            
            if new_kld_value > 0.01:
                if i > 0:
                    self.update_params(sess, self.pi_vars, pi_params)
                    self.update_params(sess, [self.pi_train_step], [current_train_step + i])
                else:
                    kld_value = new_kld_value
                    self.update_params(sess, self.pi_vars, new_pi_params)
                    self.update_params(sess, [self.pi_train_step], [current_train_step + 1])
                break
            else:
                pi_params = new_pi_params
                kld_value = new_kld_value
                self.update_params(sess, [self.pi_train_step], [current_train_step + i+1])
        
        print('Number of policy updates:', i+1)
        print(f'Learning rate: {sess.run(self.pi_lr):.2E}')

        # Update value function
        self.update_val_func(sess, batch_inputs)

        batch_summary = {self.ph_labels[ph]: values for ph, values in batch_inputs.items()}
        batch_summary['pnls'] = batch_pnls
        batch_summary['lens'] = batch_lens
        batch_summary['kld'] = kld_value
        batch_summary['r_idx'] = batch_runnners_idx
        
        return batch_summary
        
    def test_policy(self, sess, env, runner_index, max_prob=True):
        """ Test policy on single runner """
        
        # reset episode-specific variables
        obs = env.reset(runner_index=runner_index)
        done = False
        ps_list = []
        actions_list = []
        obs_list = []
        rews_list = []

        # Runner info
        runner = env.samples[runner_index].loc[env.ptr:].copy()
        price_back = runner.price_back.values
        price_lay = runner.price_lay.values

        while not done:

            obs_list.append(obs)
            
            ps = sess.run(self.pi, {self.obs_ph: np.array(obs).reshape((-1, self.obs_dim))})[0]
            if max_prob:
                act = np.argmax(ps)
            else:
                act = sess.run(self.sample_action, {self.obs_ph: np.array([obs])})[0]
            
            obs, rew, done, _ = env.step(act)
            
            ps_list.append(ps)
            actions_list.append(act)
            rews_list.append(rew)

        act_labels = []
        for a, obs in zip(actions_list, obs_list):
            act_labels.append(action_label(a, obs))

        plot_trade(np.array(ps_list), np.array(act_labels), price_back, price_lay)
        
        return ps_list, actions_list, obs_list, rews_list, act_labels, price_back, price_lay

    
# --------------------------------------------------------------------------------------------------------------
# 
#  MAIN LOOP
# 
# --------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('runners_path', type='str', help='Path to file of pre-processed races.')
    parser.add_argument('--pi_clip', type=float, default=0.2, help='Policy clip for PPO')
    parser.add_argument('--pi_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--bet_size', type=float, default=1, help='Bet size')
    parser.add_argument('--runner_index', type=int, default=-1, help='Specifies if training should occur on a single specific runner or multiple runners')
    parser.add_argument('--n_iters', type=int, default=100, help='Number of batch iterations')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for each iteration')

    args = parser.parse_args()
    if args.runner_index == -1: args.runner_index = None

    print(args)
    
    # Init environment and agent
    PI_MLP_SIZES = [32, 32, 32, 32]
    VF_MLP_SIZES = [32, 32, 32, 32]
    
    env = Env(args.runners_path)
    agent = Agent(PI_MLP_SIZES, VF_MLP_SIZES)

    session_summary = []

    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.n_iters):

            print(f'----------------------------\nEpoch: {i}')

            batch_summary = agent.train_on_batch(sess, env, args.batch_size, runner_index=args.runner_index)
            session_summary.append(batch_summary)

            print(f"mean return: {np.mean(batch_summary['pnls']):.3f}")
            print(f"return quantile: {np.quantile(batch_summary['pnls'], [.25, .5, .75])}")
            print(f"races p/batch: {len(batch_summary['pnls'])}")
            print(f"KL div: {batch_summary['kld']:.3f}")

        # Save data
        # TODO: Use h5py
        with open('sess_summary.pickle', 'wb') as f:
            pickle.dump(session_summary, f, pickle.HIGHEST_PROTOCOL)

        # Test policy on a pre-specified race for quick inspection
        ps_list, actions_list, obs_list, rews_list, act_labels, price_back, price_lay = agent.test_policy(sess, env, 1, max_prob=False)
        plt.show()

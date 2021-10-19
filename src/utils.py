import json
import os
import random
import threading
import time
from abc import ABC
from collections import deque
import cv2
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from src.model import build_actor, build_critic


class SupervisedDataset:

    def __init__(self, data_path='/data/', train=True, pad=True, gray=True):
        self.train = train
        if train:
            anno_path = os.path.join(data_path, 'frames/training_annotations.json')
        else:
            anno_path = os.path.join(data_path, 'frames/testing_annotations.json')
        self.frames_path = os.path.join(data_path, 'frames/')
        self.data_path = data_path
        self.annotations = json.load(open(anno_path, 'r'))
        self.pad = pad
        self.gray = gray

        # Collect items
        def fn(key, item):
            base_frame_num, num, target = item
            return f"{key}_{base_frame_num}.png", f"{key}_{num}.png", target

        base_paths, paths, targets = [], [], []
        for key, items in self.annotations.items():
            for item in items:
                a, b, c = fn(key, item)
                base_paths.append(a)
                paths.append(b)
                targets.append(c)
        self.size = len(targets)
        self.dataset = tf.data.Dataset.from_tensor_slices((base_paths, paths, targets)).repeat()
        if train:
            self.dataset = self.dataset.shuffle(self.size)
        print(self.__get_summary(targets))

    def __get_summary(self, targets):
        n_frames = len(targets)
        n_frames1 = len([x for x in targets if x == 1])
        n_frames0 = len([x for x in targets if x == 0])
        return f"Frames: {n_frames} | IRelevant : {n_frames1} | Relevant : {n_frames0}"

    def map_fn(self, base_paths, paths, target):
        im1 = self.load_image(base_paths)
        im2 = self.load_image(paths)
        image = im2 - im1
        target = tf.one_hot(target, 2)
        return (image, -1), target

    def load_image(self, name):
        image = tf.io.read_file(self.frames_path + "/" + name)
        image = tf.image.decode_image(image, expand_animations=False)
        if self.gray:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32)
        return image

    def get_dataset(self, batch_size=32, prefetch=10):
        dataset = self.dataset.map(self.map_fn, num_parallel_calls=4)
        dataset = dataset.batch(batch_size).prefetch(prefetch)
        return dataset, self.size // batch_size

    def __len__(self):
        return self.size


def draw_bboxes(image, bboxes):
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    for i in range(len(bboxes)):
        ymin, xmin, ymax, xmax = bboxes[i]
        start_point = (int(ymin), int(xmin))
        end_point = (int(ymax), int(xmax))
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # cv2.putText(image, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
    return image


def faster_rcnn_predict(image, device=None, target_class=None):
    if not hasattr(faster_rcnn_predict, 'data'):
        from detectron2.utils.logger import setup_logger
        setup_logger()
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        import torch

        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.DEVICE = device if device else cfg.MODEL.DEVICE
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        model = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        # save
        faster_rcnn_predict.data = model, metadata
    model, metadata = faster_rcnn_predict.data

    pred = model(image)
    classes = pred['instances'].pred_classes.cpu().numpy()
    bboxes = pred['instances'].pred_boxes.tensor.cpu().numpy()

    if target_class is not None:
        idx = classes == target_class
        # classes = classes[idx]
        bboxes = bboxes[idx]
        return bboxes

    return classes, bboxes


clipping_val = 0.1
critic_discount = 0.5
entropy_beta = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def compare_bboxes(boxes, base_bboxes, iou_tolerance=0.1):
    if boxes.shape[0] != base_bboxes.shape[0]:
        return False
    for i in range(boxes.shape[0]):
        bb1 = boxes[i]
        equ = False
        for j in range(base_bboxes.shape[0]):
            bb2 = base_bboxes[j]
            iou = batch_iou(bb1[np.newaxis], bb2[np.newaxis])
            iou = 1 - iou
            if np.all(iou <= iou_tolerance):
                equ = True
                break
        if not equ:
            return False
    return True


class EpisodesPool:

    def __init__(self, window_size=100, pool_size=1000):
        self.pool = deque(maxlen=pool_size)
        self.buffer = []
        self.window_size = window_size

    def put(self, image, bboxes):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
        image = image.reshape((112, 112, 1))
        self.buffer.append((image, bboxes))
        if len(self.buffer) >= self.window_size:
            states, actions = self.process_buffer()
            arr = np.array(actions)
            # if ~(np.all(arr == 1) or np.all(arr == 0)):
            self.pool.append((states, actions))
            self.buffer = []

    def process_buffer(self):
        base_bbox = None
        base_frame = None
        states = []
        actions = []
        for image, bboxes in self.buffer:
            if base_frame is None:
                base_frame = image
                base_bbox = bboxes
            if compare_bboxes(bboxes, base_bbox, iou_tolerance=0.1):
                action = 1
            else:
                action = 0
                base_frame = image
                base_bbox = bboxes
            state = (image - base_frame).astype(np.int16)[np.newaxis]
            states.append(state)
            actions.append(action)
        return states, actions

    def sample(self):
        while True:
            try:
                data = random.choice(self.pool)
                return data[0][:], data[1][:]
            except:
                print("Empty Pool")
                time.sleep(10)

    def __len__(self):
        return len(self.pool)


class FilterEnv(gym.Env, ABC):
    def __init__(self, pool: EpisodesPool):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0.0, high=1.0, shape=(112, 112, 1), dtype=np.float32),
            gym.spaces.Box(low=-1., high=1.0, shape=(), dtype=np.float32),
        ))

        self.pool = pool
        self.states = []
        self.actions = []
        self.state = None, None
        self.true_action = None

    def step(self, action):
        reward = 1 if self.true_action == action else -1
        done = len(self.states) == 0
        if not done:
            frame, self.true_action = self.states.pop(0), self.actions.pop(0)
            self.state = frame, np.array([action, ])
        info = {"true_action": self.true_action}
        return self.state, reward, done, info

    def reset(self):
        self.states, self.actions = self.pool.sample()
        frame, self.true_action = self.states.pop(0), self.actions.pop(0)
        prev_action = np.array([-1, ])
        self.state = frame, prev_action
        return self.state


class OnlinePPOAdapter:

    def __init__(self, pool: EpisodesPool, camera_name, deterministic=False,
                 models_path="../models", log_dir='../logs', batch=1):
        self.batch = batch
        self.pool = pool
        self.env = FilterEnv(pool)
        self.actor = build_actor()
        self.critic_warmup_steps = 100
        self.camera_name = camera_name
        self.max_steps = 10000
        self.gamma = 0.99
        self.deterministic = deterministic
        self.steps = 1
        self.sum_returns = 0.0
        self.best_returns = float('-inf')
        self.best_steps = 1
        self.sum_advantages = 0.0
        self.sum_loss_actor = 0.0
        self.sum_loss_critic = 0.0
        self.thread = threading.Thread(target=OnlinePPOAdapter.run, args=(self,))

        self.actor_optimizer = keras.optimizers.Adam(1e-6)
        self.critic_optimizer = keras.optimizers.Adam(1e-5)
        log_path = os.path.join(log_dir, "rl", camera_name)
        os.makedirs(log_path, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_path)

        # Load Weights
        pretrained_path = os.path.join(models_path, "pretrained_sf_oh.h5")
        self.critic_path = os.path.join(models_path, f"critic_{camera_name.lower()}.h5")
        self.actor_path = os.path.join(models_path, f"actor_{camera_name.lower()}.h5")
        if os.path.exists(self.critic_path):
            print(f"{camera_name} : Loading critic from {self.critic_path}...")
            self.critic = build_critic(self.critic_path, from_actor=False)
        else:
            print(f"{camera_name} : Loading critic from {pretrained_path}...")
            self.critic = build_critic(pretrained_path)
        if os.path.exists(self.actor_path):
            print(f"{camera_name} : Loading actor from {self.actor_path}...")
            self.critic = build_critic(self.actor_path)
        else:
            print(f"{camera_name} : Loading actor from {pretrained_path}...")
            self.critic = build_critic(pretrained_path)

    def start(self):
        self.thread.start()

    @tf.function
    def update_actor(self, actor_data):
        X, batch_advantage, batch_actions_probs, batch_actions = actor_data

        with tf.GradientTape() as tape:
            predictions = self.actor(X)
            loss = self.ppo_loss(batch_advantage, predictions, batch_actions_probs, batch_actions)
        grads = tape.gradient(loss, self.actor.trainable_weights)

        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        return loss

    def ppo_loss(self, batch_advantage, predictions, batch_actions_probs, batch_actions):
        old_actions = tf.one_hot(batch_actions, 2)

        new_actions = tf.argmax(predictions, -1)
        new_actions = tf.one_hot(new_actions, 2)

        log_predictions = tf.math.log(predictions)
        log_batch_actions_probs = tf.math.log(batch_actions_probs)

        prob_new = log_predictions * new_actions
        prob_old = log_batch_actions_probs * old_actions
        ratio = tf.math.exp(prob_new - prob_old)

        surr1 = ratio * batch_advantage
        surr2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * batch_advantage
        actor_loss = - K.mean(K.minimum(surr1, surr2))

        entropy = entropy_beta * K.mean(predictions * log_predictions)
        return actor_loss - entropy

    @tf.function
    def update_critic(self, critic_data):
        X, batch_returns = critic_data

        with tf.GradientTape() as tape:
            pred = self.critic(X, training=True)
            loss = tf.reduce_mean(keras.losses.mse(batch_returns, pred)) * critic_discount

        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        return loss

    def run(self):
        while self.steps <= self.max_steps:
            actor_data, critic_data = self.collect_episode_batch(self.batch)
            # Update actor and critic
            if self.critic_warmup_steps <= 0:
                self.sum_loss_actor += self.update_actor(actor_data).numpy()
            else:
                self.steps = 1
                self.sum_returns = 0.0
                self.best_returns = float('-inf')
                self.best_steps = 1
                self.sum_advantages = 0.0
                self.sum_loss_actor = 0.0
                self.sum_loss_critic = 0.0
                self.critic_warmup_steps -= 1

            self.sum_loss_critic += self.update_critic(critic_data).numpy()
            msg = self.save_logs()
            print(msg)
            # save best agent
            if (self.sum_returns / self.steps) >= (self.best_returns / self.best_steps):
                if self.steps > 1000:
                    self.actor.save_weights(self.actor_path)
                    self.critic.save_weights(self.critic_path)
                    self.best_returns = self.sum_returns
                    self.best_steps = self.steps
                    print(f"{self.camera_name} : Saving best agent ....")

            self.steps += 1

    def collect_episode_batch(self, n=2):
        batch_images = []
        batch_prev_actions = []
        batch_actions_probs = []
        batch_actions = []
        batch_returns = []
        batch_advantage = []
        for episode in range(n):
            (images, prev_actions), actions_probs, actions, rewards, dones, values = self.collect_episode()

            returns = self.compute_gae(values, dones, rewards)
            advantage = returns - values
            advantage = self.normalize(advantage)

            # data collect
            batch_advantage.append(advantage)
            batch_returns.append(returns)
            batch_images.append(images)
            batch_prev_actions.append(prev_actions)
            batch_actions_probs.append(actions_probs)
            batch_actions.append(actions)

        batch_images = np.concatenate(batch_images, 0)
        batch_prev_actions = np.concatenate(batch_prev_actions, 0)
        batch_actions_probs = np.concatenate(batch_actions_probs, 0)
        batch_actions = np.concatenate(batch_actions, 0)
        batch_returns = np.concatenate(batch_returns, 0)
        batch_advantage = np.concatenate(batch_advantage, 0)

        # Save logs
        self.sum_returns += batch_returns.sum()
        self.sum_advantages += batch_advantage.sum()

        X = (batch_images, batch_prev_actions)
        actor_data = X, batch_advantage, batch_actions_probs, batch_actions
        critic_data = X, batch_returns
        return actor_data, critic_data

    def save_logs(self):
        returns = self.sum_returns / self.steps
        with self.writer.as_default():
            tf.summary.scalar("returns", returns, self.steps)
            tf.summary.scalar("advantages", self.sum_advantages / self.steps, self.steps)
            tf.summary.scalar("loss_actor", self.sum_loss_actor / self.steps, self.steps)
            tf.summary.scalar("loss_critic", self.sum_loss_critic / self.steps, self.steps)

        msg = f">>>CameraID: {self.camera_name} | " \
              f"Steps: {self.steps}/{self.max_steps} | " \
              f"Critic WarmUp: {self.critic_warmup_steps} | " \
              f"Returns: {returns:.2f} | " \
              f"Poll Size: {len(self.pool)} | " \
              f"Best Return: {(self.best_returns / self.best_steps):.2f}"

        return msg

    def collect_episode(self, ):
        done = False
        obs = self.env.reset()

        frames = []
        prev_actions = []
        rewards = []
        actions = []
        actions_probs = []
        dones = []
        true_actions = []
        while not done:
            action_prob = self.actor(obs).numpy()[0]
            if self.deterministic:
                action = np.argmax(action_prob, -1)
            else:
                action = np.random.choice([0, 1], 1, p=action_prob)[0]
            new_obs, reward, done, info = self.env.step(action)
            true_actions.append(info['true_action'])
            frames.append(obs[0])
            prev_actions.append(obs[1])
            rewards.append(reward)
            actions.append(action)
            actions_probs.append(action_prob)
            dones.append(int(done))
            obs = new_obs
        images = np.concatenate(frames, axis=0)
        prev_actions = np.concatenate(prev_actions, axis=0)
        true_actions = np.array(true_actions)
        # rewards = np.array(rewards)
        rewards = self.balanced_rewards(true_actions, actions)
        actions_probs = np.array(actions_probs)
        actions = np.array(actions)
        dones = np.array(dones)
        values = self.critic((images, prev_actions)).numpy()
        return (images, prev_actions), actions_probs, actions, rewards, dones, values

    def balanced_rewards(self, true_actions, actions):
        max_reward = 50
        factor1 = sum(true_actions == 1) / max_reward
        factor0 = sum(true_actions == 0) / max_reward
        rewards = []
        for true_action, action in zip(true_actions, actions):
            if true_action == 1:
                factor = factor1
            if true_action == 0:
                factor = factor0
            rewards.append(factor if true_action == action else -factor)
        return np.array(rewards)

    def compute_gae(self, values, masks, rewards, gamma=GAMMA, lam=GAE_LAMBDA):
        values = list(values) + [0]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return np.array(returns)

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

import numpy as np

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
class PIDController(BaseController):
  def __init__(self, Kp=0.1, Ki=0.01, Kd=0.1):
    self.integral = 0
    self.prev_error = 0
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd

  def update(self, target_lataccel, current_lataccel, state):
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error
    self.prev_error = error
    steer_action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    return steer_action
class LQRController(BaseController):
  def __init__(self, A, B, Q, R):
    self.A = np.array(A)
    self.B = np.array(B)
    self.Q = np.array(Q)
    self.R = np.array(R)
    self.K = self.compute_lqr_gain()

  def compute_lqr_gain(self):
    # Solve the continuous time lqr controller.
    # x_dot = Ax + Bu
    # cost = integral x.T*Q*x + u.T*R*u
    X = np.matrix(np.linalg.solve(self.A.T, self.Q + self.A.T * X * self.A - self.A.T * X * self.B * np.linalg.inv(self.R + self.B.T * X * self.B) * self.B.T * X * self.A))
    K = np.linalg.inv(self.R + self.B.T * X * self.B) * self.B.T * X * self.A
    return np.array(K)

  def update(self, target_lataccel, current_lataccel, state):
    # State feedback u = -Kx
    x = np.array([target_lataccel - current_lataccel])  # state difference
    u = -self.K.dot(x)
    return u[0]  # Return the first element as the control action


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'lqr': LQRController
}

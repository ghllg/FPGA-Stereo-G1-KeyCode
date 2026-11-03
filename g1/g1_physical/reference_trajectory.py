"""Smooth recovered upper-body references while preserving support motion."""
import numpy as np
from physical import ReferenceBuffer


class SmoothReferenceBuffer(ReferenceBuffer):
    # A regression window, not an added playback delay. The existing future
    # guard still requires every requested time to be inside received samples.
    sigma = .45

    def __init__(self, model, interpolation='linear'):
        super().__init__(model, interpolation)
        self.indices = np.array([i for i in range(model.njnt-1)
                                 if any(part in model.joint(i+1).name
                                        for part in ('waist', 'shoulder', 'elbow', 'wrist'))])

    def sample(self, timestamp):
        q, velocity = super().sample(timestamp)
        if len(self.times) < 4:
            return q, velocity
        x = (np.array(self.times)-timestamp)/self.sigma
        poses = np.array(self.poses)
        weights = np.exp(-.5*x*x)
        values = poses[:,7+self.indices]
        design = np.c_[np.ones(len(x)), x, x*x]
        normal = design.T @ (weights[:,None]*design)
        if np.linalg.cond(normal) > 1e8:
            return q, velocity
        coef = np.linalg.solve(normal, design.T @ (weights[:,None]*values))
        residual = values-design@coef
        # Differentiating the moving regression includes the kernel derivative.
        # The polynomial slope alone is not the derivative of sample(timestamp).
        change = np.linalg.solve(normal, design.T @ ((weights*x)[:,None]*residual))[0]
        # Filtering floating-base/leg references can delay swing and support
        # changes. Keep their original interpolated positions and velocities.
        values = coef[0]
        speed = (coef[1]+change)/self.sigma
        limited = self.model.jnt_limited[1+self.indices].astype(bool)
        bounds = self.model.jnt_range[1+self.indices]
        outside = limited & ((values < bounds[:,0]) | (values > bounds[:,1]))
        values[limited] = np.clip(values[limited], bounds[limited,0], bounds[limited,1])
        speed[outside] = 0.
        q[7+self.indices] = values
        velocity[self.indices] = speed
        return q, velocity

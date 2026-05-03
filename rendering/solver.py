import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk

@jdc.jit
def _solve_ik_jax_batched(
    robot: pk.Robot,
    target_link_index: jax.Array,        # scalar
    target_wxyz: jax.Array,               # (N, 4)
    target_position: jax.Array,           # (N, 3)
) -> jax.Array:                           # (N, num_joints)
    
    def solve_single(wxyz, pos):
        joint_var = robot.joint_var_cls(0)
        variables = [joint_var]
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(wxyz), pos
                ),
                target_link_index,
                pos_weight=50.0,
                ori_weight=10.0,
            ),
            pk.costs.limit_constraint(robot, joint_var),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
        )
        return sol[joint_var]

    return jax.vmap(solve_single)(target_wxyz, target_position)
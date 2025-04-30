"""
This script launches a locomotion MPC scheme which solves repeatedly an 
optimal control problem based on the full dynamics model of the humanoid robot Talos. 
The contacts forces are modeled as 6D wrenches. 
"""

import numpy as np
import aligator
import pinocchio as pin
from bullet_robot import BulletRobot
import time

from talos_utils import (
    loadTalos,
    URDF_FILENAME,
    modelPath,
    shapeState,
    footTrajectory,
    update_timings,
    save_trajectory
)

from aligator import (manifolds, 
                    dynamics, 
                    constraints,)

# ============================
# INITIAL SETUP
# ============================
print("FULL DYNAMIICS MODEL:\n\
      x = [q(29), q_dot(28)]\n\
      u = [tau(23)]")

print("============================\n\
      INITIAL SETUP\n\
      ============================\n")

# Load Talos Model
rmodelComplete, rmodel, qComplete, q0 = loadTalos()
rdata = rmodel.createData()
print("TALOS MODEL LOADED:")
print(rmodel.joints[:].tolist())
nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
gravity = -9.81

# Robot Parameters
mass = pin.computeTotalMass(rmodel)
mu = 0.8
Lfoot = 0.1
Wfoot = 0.075

# Torque (input) limits
umax = rmodel.effortLimit[6:]
umin = -umax




print("FULL DYNAMIICS MODEL:\n\
      x = [q("+str(nq)+"), q_dot("+str(nv)+")]\n\
      u = [tau("+str(nu)+")]")

# Get IDs for frames and joints
FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname)
    for fname in ["left_sole_link", "right_sole_link"]
}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}
LF_id = rmodel.getFrameId("left_sole_link")
RF_id = rmodel.getFrameId("right_sole_link")
root_id = rmodel.getFrameId("root_joint")
LF_placement = rdata.oMf[LF_id]
RF_placement = rdata.oMf[RF_id]
LF_init = rdata.oMf[LF_id].copy()
RF_init = rdata.oMf[RF_id].copy()

# Controlled Joints
controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [rmodelComplete.getJointId(name_joint) for name_joint in controlled_joints[1:]]
print("\nControlled Joints: "+str(len(controlled_joints))+"; n_u: "+str(nu))

# Initialize Robot in Half Sitting Position
q0 = rmodel.referenceConfigurations["half_sitting"]
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

# ============================
# INITIALIZE SIMULATION
# ============================
print("\n============================\n\
      INITIALIZE SIMULATION\n\
      ============================\n")
device = BulletRobot(controlled_joints,
                        modelPath,
                        URDF_FILENAME,
                        1e-3,
                        rmodelComplete)
device.initializeJoints(qComplete)
device.changeCamera(1., 30, -10, [1., -0.6, 1.])
q_current, v_current = device.measureState()

space = manifolds.MultibodyPhaseSpace(rmodel)

# Initial States/inputs
x0 = np.concatenate((q0, np.zeros(nv)))

# Actuation Matrix S
act_matrix = np.eye(nv, nu, -6)
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel,rdata,x0[:nq])

tau = np.sqrt(com0[2] / 9.81) # Time constant for DCM

# ??? 
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 1)

frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)
v_ref = pin.Motion()
v_ref.np[:] = 0.0


constraint_models = []
constraint_datas = []
for fname, fid in FOOT_FRAME_IDS.items():
    joint_id = FOOT_JOINT_IDS[fname]
    pl1 = rmodel.frames[fid].placement
    pl2 = rdata.oMf[fid]
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        rmodel,
        joint_id,
        pl1,
        0,
        pl2,
        pin.LOCAL,
    )
    cm.corrector.Kp[:] = (1, 1, 10, 1, 1, 1) #(0, 0, 10, 0, 0, 0)
    cm.corrector.Kd[:] = (50, 50, 50, 50, 50, 50)
    cm.name = fname
    constraint_models.append(cm)
    constraint_datas.append(cm.createData())


def create_dynamics(stage_space, cs):
    dyn_model = None
    if cs[0] and not(cs[1]):
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, [constraint_models[0]], prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    elif cs[1] and not(cs[0]):
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, [constraint_models[1]], prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    else:
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, constraint_models, prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    return dyn_model 


"""
MPC WEIGHTS

RUNNING COST:
Lt(x,u) = |x−x∗|2 + |u−u∗(t)|2 + |p(x)−p∗(t)|2 + |λ(x,u)−λ∗(t)|2

"""

# State weight |x−x∗|2(Dx)
w_x = np.array([
    0, 0, 0, 100, 100, 100, # Base pos/ori
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Left leg
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Right leg
    10, 10, # Torso
    1, 1, 1, 1, # Left arm
    1, 1, 1, 1, # Right arm
    1, 1, 1, 1, 1, 1, # Base pos/ori vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01, # Left leg vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01, # Right leg vel
    10, 10, # Torso vel
    1, 1, 1, 1, # Left arm vel
    1, 1, 1, 1, # Right arm vel
]) 
w_x = np.diag(w_x) * 1

# Input weight |u−u∗(t)|2 (Du)
w_u = np.eye(nu) * 1e-4

#
w_LFRF = 2000
w_com = 100 * np.ones(3)
w_com = np.diag(w_com) 

w_cent_lin = np.array([0.0,0.0,10]) * 0
w_cent_ang = np.array([0.0,0.0,10])
w_cent = np.diag(np.concatenate((w_cent_lin,w_cent_ang)))

w_forces_lin = np.array([0.0001, 0.0001, 0.0001])
w_forces_ang = np.ones(3) * 0.0001
w_forces = np.diag(np.concatenate((w_forces_lin,w_forces_ang)))


# Create MPC stage at time t
def createStage(cs, cs_previous, LF_target, RF_target, LF_force, RF_force):
    stage_rmodel = rmodel.copy()
    stage_space = manifolds.MultibodyPhaseSpace(stage_rmodel)

    frame_vel_LF = aligator.FrameVelocityResidual(stage_space.ndx, nu, rmodel, v_ref, LF_id, pin.LOCAL)
    frame_vel_RF = aligator.FrameVelocityResidual(stage_space.ndx, nu, rmodel, v_ref, RF_id, pin.LOCAL)


    cent_mom = aligator.CentroidalMomentumResidual(
        stage_space.ndx, nu, stage_rmodel, np.zeros(6)
    )

    frame_fn_LF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, rmodel, LF_target, LF_id)
    frame_fn_RF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, rmodel, RF_target, RF_id)
    frame_cs_RF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, rmodel, RF_target.translation, RF_id)[2]
    frame_cs_LF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, rmodel, LF_target.translation, LF_id)[2]
    frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)[2]

    # RUNNING COST
    rcost = aligator.CostStack(stage_space, nu)

    # State cost |x−x∗|2(Dx) - 0
    rcost.addCost(aligator.QuadraticStateCost(stage_space, nu, x0, w_x))
    
    # Input cost |x−x∗|2(Dx) - 1
    rcost.addCost(aligator.QuadraticControlCost(stage_space, u0, w_u))
    
    w_LF = np.zeros((6,6))
    w_RF = np.zeros((6,6))
    if cs[0]: # and not(cs[1]):
        w_RF = w_LFRF * np.eye(6)
    if cs[1]: # and not(cs[0]):
        w_LF = w_LFRF * np.eye(6)

    # - 2
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_LF, w_LF))
    # - 3
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_RF, w_RF))
    # - 4
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, cent_mom, w_cent))
    #rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_com, w_com))
    
    # |λ(x,u)−λ∗(t)|2 wrench constraint depending on n_contact points (L, R, L+R)
    if cs[0] and cs[1]: # L + R double contact
        frame_force_LF = aligator.ContactForceResidual(
            stage_space.ndx, rmodel, act_matrix, constraint_models, prox_settings, LF_force, "left_sole_link") 
        frame_force_RF = aligator.ContactForceResidual(
            stage_space.ndx, rmodel, act_matrix, constraint_models, prox_settings, RF_force, "right_sole_link") 
        rcost.addCost(aligator.QuadraticResidualCost(stage_space,frame_force_LF, w_forces))
        rcost.addCost(aligator.QuadraticResidualCost(stage_space,frame_force_RF, w_forces))
    elif cs[0]: # L single contact
        frame_force_LF = aligator.ContactForceResidual(
            stage_space.ndx, rmodel, act_matrix, [constraint_models[0]], prox_settings, LF_force, "left_sole_link")
        rcost.addCost(aligator.QuadraticResidualCost(stage_space,frame_force_LF, w_forces))
    elif cs[1]: # R single contact
        frame_force_RF = aligator.ContactForceResidual(
            stage_space.ndx, rmodel, act_matrix, [constraint_models[1]], prox_settings, RF_force, "right_sole_link")
        rcost.addCost(aligator.QuadraticResidualCost(stage_space,frame_force_RF, w_forces))


    stm = aligator.StageModel(rcost, create_dynamics(stage_space, cs))
    
    """ CONSTRAINTS:

    gl_i(xi,ui) = [Xjxi−ql, qu−Xjxi, # joint limita
                    τu−ui, τu + ui] # torque limits

    
    """
    # Torque (input) limits 
    ctrl_fn = aligator.ControlErrorResidual(stage_space.ndx, np.zeros(nu))
    #stm.addConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax))

    # Joint Limits
    state_fn = aligator.StateErrorResidual(stage_space, nu, stage_space.neutral())[6:nv]
    #stm.addConstraint(state_fn, constraints.BoxConstraint(-rmodel.upperPositionLimit[7:], -rmodel.lowerPositionLimit[7:]))
    
    # Friction Cone Constraint based on contact state (L,R,L+R)
    if cs[0] and not(cs[1]): # L single contact
        frame_cone_LF_const = aligator.MultibodyWrenchConeResidual(
            stage_space.ndx, rmodel, act_matrix, [constraint_models[0]], prox_settings, "left_sole_link", mu, Lfoot, Wfoot)
        stm.addConstraint(frame_cone_LF_const, constraints.NegativeOrthant())
    elif cs[1] and not(cs[0]): # R single contact
        frame_cone_RF_const = aligator.MultibodyWrenchConeResidual(
            stage_space.ndx, rmodel, act_matrix, [constraint_models[1]], prox_settings, "right_sole_link", mu, Lfoot, Wfoot)
        stm.addConstraint(frame_cone_RF_const, constraints.NegativeOrthant())
    else: # L+R double contact
        frame_cone_LF_const = aligator.MultibodyWrenchConeResidual(
            stage_space.ndx, rmodel, act_matrix, constraint_models, prox_settings, "left_sole_link", mu, Lfoot, Wfoot)
        stm.addConstraint(frame_cone_LF_const, constraints.NegativeOrthant())
        frame_cone_RF_const = aligator.MultibodyWrenchConeResidual(
            stage_space.ndx, rmodel, act_matrix, constraint_models, prox_settings, "right_sole_link", mu, Lfoot, Wfoot)
        stm.addConstraint(frame_cone_RF_const, constraints.NegativeOrthant())
    """ if cs[1] and not(cs_previous[1]):
        stm.addConstraint(frame_cs_RF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_vel_RF, constraints.EqualityConstraintSet())
    if cs[0] and not(cs_previous[0]):
        stm.addConstraint(frame_cs_LF, constraints.EqualityConstraintSet()) 
        stm.addConstraint(frame_vel_LF, constraints.EqualityConstraintSet())  """
    return stm

term_cost = aligator.CostStack(space, nu)
"""
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x * 1))
cent_mom = aligator.CentroidalMomentumResidual(
    space.ndx, nu, rmodel, np.zeros(6)
)
frame_fn_LF = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, LF_placement, LF_id)
frame_fn_RF = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, RF_placement, RF_id)
term_cost.addCost(aligator.QuadraticResidualCost(space, cent_mom, w_cent))
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_fn_LF, w_LFRF * np.eye(6)))
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_fn_RF, w_LFRF * np.eye(6)))
"""

""" Define gait and time parameters"""
T_ds = 20
T_ss = 80

"""
Define a full step sequence as 100 'instants'
[INITIAL STAND]
20 -> L - R

[First Step]
80 -> L 
20 -> L - R

[Second Step]
80 -> R
20 -> L - R

[ADD 2 "step times" of stillness]
200 -> L - R
"""
dt = 0.01
nsteps = 100
Nsimu = int(dt / 0.001)

""" Define contact sequence throughout horizon"""
total_steps = 3
contact_phases = [[True,True]] * T_ds
for s in range(total_steps):
    contact_phases += [[True,False]] * T_ss + \
                      [[True,True]] * T_ds + \
                      [[False,True]] * T_ss + \
                      [[True,True]] * T_ds 

"""
contact_phases += [[True,False]] * T_ss + \
                  [[True,True]] * T_ds
"""

contact_phases += [[True,True]] * nsteps * 2

takeoff_RFs = []
takeoff_LFs = []
land_RFs = []
land_LFs = []
for i in range(1, len(contact_phases)):
    if contact_phases[i] == [True, False] and contact_phases[i - 1] == [True, True]:
        takeoff_RFs.append(i + nsteps)
    elif contact_phases[i] == [False, True] and contact_phases[i - 1] == [True, True]:
        takeoff_LFs.append(i + nsteps)
    elif contact_phases[i] == [True, True] and contact_phases[i - 1] == [True, False]:
        land_RFs.append(i + nsteps)
    elif contact_phases[i] == [True, True] and contact_phases[i - 1] == [False, True]:
        land_LFs.append(i + nsteps)

print("Takeoff R: "+str(takeoff_RFs)+
      "\nTakeoff L: "+str(takeoff_LFs)+
      "\nLanding R: "+str(land_RFs)+
      "\nLanding L: "+str(land_LFs))

Tmpc = len(contact_phases)
print("Time MPC: "+str(Tmpc))

f_full = -mass * gravity
f_half = -mass * gravity / 2.

LF_force_refs = []
RF_force_refs = []

"""
REDISTRIBUTE FORCES LIKE THIS:

PHASE I:
Double Contact -> Single Contact
L : (50% -> 100%) Gravity Comp.
R : (50% -> 0%) Gravity Comp.

PHASE II:
Single contact
L : 100% Garavity Comp.
R :   0% Garavity Comp.

PHASE III:
Weight shift: L -> R
L : (100% -> 0%) Gravity Comp.
R : (0% -> 100%) Gravity Comp.

PHASE IV:
Single contact
L :   0% Garavity Comp.
R : 100% Garavity Comp.

IF MORE STEPS, REPEAT WITH MODIFIED PHASE I:
Weight shift: R -> L
L : (0% -> 100%) Gravity Comp.
R : (100% -> 0%) Gravity Comp.
_______________________________
FINALLY, AFTER ALL TOTAL_STEPS:

PHASE V:
Re-balance
L : (  0% -> 50%) Gravity Comp.
R : (100% -> 50%) Gravity Comp.

PHASE VI:
Double contact
L : 50% Garavity Comp.
R : 50% Garavity Comp.

"""
for i in range(total_steps):
    # PHASE I:
    for j in range(T_ds):
        LF_force_ref = np.zeros(6)
        RF_force_ref = np.zeros(6)
        if i == 0:
            LF_force_ref[2] = f_full * j / T_ds + f_half * (T_ds - j) / T_ds
            RF_force_ref[2] = f_half * (T_ds - j) / T_ds 
        else:
            LF_force_ref[2] = f_full * (j + 1) / T_ds
            RF_force_ref[2] = f_full * (T_ds - j) / T_ds 
        
        LF_force_refs.append(LF_force_ref)
        RF_force_refs.append(RF_force_ref)

    # PHASE II    
    for j in range(T_ss):
        LF_force_ref = np.zeros(6)
        RF_force_ref = np.zeros(6)
        LF_force_ref[2] = f_full
        LF_force_refs.append(LF_force_ref)
        RF_force_refs.append(RF_force_ref)

    # PHASE III
    for j in range(T_ds):
        LF_force_ref = np.zeros(6)
        RF_force_ref = np.zeros(6)
        LF_force_ref[2] = f_full * (T_ds - j) / T_ds
        RF_force_ref[2] = f_full * (j + 1) / T_ds 
        LF_force_refs.append(LF_force_ref)
        RF_force_refs.append(RF_force_ref)

    # PHASE IV
    for j in range(T_ss):
        LF_force_ref = np.zeros(6)
        RF_force_ref = np.zeros(6)
        RF_force_ref[2] = f_full
        LF_force_refs.append(LF_force_ref)
        RF_force_refs.append(RF_force_ref)
"""
for j in range(len(LF_force_refs)):
    print("L: "+str(round(LF_force_refs[j][2]/f_full,3)) +" - R: "+str(round(RF_force_refs[j][2]/f_full,2)))
"""

# PHASE V
for j in range(T_ds):
    LF_force_ref = np.zeros(6)
    RF_force_ref = np.zeros(6)
    LF_force_ref[2] = f_half * (j + 1) / float(T_ds)
    RF_force_ref[2] = f_full * (T_ds - j) / float(T_ds) + f_half * j / float(T_ds)
    LF_force_refs.append(LF_force_ref)
    RF_force_refs.append(RF_force_ref)

# PHASE VI
"""
for j in range(T_ss):
    LF_force_ref = np.zeros(6)
    RF_force_ref = np.zeros(6)
    LF_force_ref[2] = f_full
    LF_force_refs.append(LF_force_ref)
    RF_force_refs.append(RF_force_ref)

for j in range(T_ds):
    LF_force_ref = np.zeros(6)
    RF_force_ref = np.zeros(6)
    RF_force_ref[2] = f_half * (j + 1) / float(T_ds)
    LF_force_ref[2] = f_full * (T_ds - j) / float(T_ds) + f_half * j / float(T_ds)
    LF_force_refs.append(LF_force_ref)
    RF_force_refs.append(RF_force_ref)
"""
for j in range(nsteps * 2):
    LF_force_ref = np.zeros(6)
    RF_force_ref = np.zeros(6)
    LF_force_ref[2] = f_half 
    RF_force_ref[2] = f_half 
    LF_force_refs.append(LF_force_ref)
    RF_force_refs.append(RF_force_ref)

""" Define feet trajectory """
swing_apex = 0.25
x_forward = 0.3
y_forward = 0.0
foot_yaw = 0
y_gap = 0.18
z_height = 0.0

foottraj = footTrajectory(
    rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), T_ss, T_ds, nsteps, swing_apex, x_forward, y_forward, foot_yaw, y_gap, z_height
)

""" Create the optimal problem and the full horizon """
stages = [createStage(contact_phases[0],contact_phases[0], LF_placement.copy(), RF_placement.copy(), LF_force_refs[0].copy(), RF_force_refs[0].copy())] * nsteps

stages_full = [createStage(
    contact_phases[0],contact_phases[0], LF_placement.copy(), RF_placement.copy(), LF_force_refs[0].copy(), RF_force_refs[0].copy())]
for i in range(1,Tmpc):
    stages_full.append(createStage(
        contact_phases[i],contact_phases[i-1], LF_placement.copy(), RF_placement.copy(), LF_force_refs[i], RF_force_refs[i]))

stages_full_data = []
for i in range(Tmpc):
    stages_full_data.append(stages_full[i].createData())

problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Parametrize the solver"""

TOL = 1e-5
mu_init = 1e-8 

max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init)# , verbose=verbose)
#solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
#print("LDLT algo choice:", solver.ldlt_algo_choice)
solver.linear_solver_choice = aligator.LQ_SOLVER_SERIAL #LQ_SOLVER_SERIAL 
solver.force_initial_condition = True
solver.setNumThreads(1)
solver.max_iters = max_iters

solver.setup(problem)

us_init = [np.zeros(nu)] * nsteps
xs_init = [x0] * (nsteps + 1) 

solver.run(
    problem,
    xs_init,
    us_init,
)

workspace = solver.workspace
results = solver.results
print(results)

xs = results.xs.tolist().copy()
us = results.us.tolist().copy()
K_feedback = results.controlFeedbacks()[0]

solver.max_iters = 1

x_measured = shapeState(q_current, 
                        v_current, 
                        nq, 
                        nq + nv, 
                        controlled_ids)

force_left = []
force_right = []
torque_left = []
torque_right = []
LF_measured = []
RF_measured = []
LF_references = []
RF_references = []
x_multibody = []
u_multibody = []
com_measured = []
solve_time = []
L_measured = []

device.showTargetToTrack(LF_placement, RF_placement)
lowlevel_time = 0
time_computation = 0.05

fd = 300
theta = 6 * np.pi / 4
f_disturbance = [np.cos(theta)* fd, np.sin(theta) * fd, 0]

x_measured_prev = xs[0].copy()
for t in range(Tmpc):
    print("Time " + str(t))

    takeoff_RF, takeoff_LF, land_RF, land_LF = update_timings(
        land_LFs, land_RFs, takeoff_LFs, takeoff_RFs
    )

    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    if land_RF == -1 and takeoff_RF == -1:
        foottraj.updateForward(0, 0, y_gap, y_forward, 0, 0, swing_apex)


    LF_refs, RF_refs = foottraj.updateTrajectory(
        takeoff_RF, takeoff_LF, land_RF, land_LF, rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()
    )
    
    for j in range(nsteps):
        problem.stages[j].cost.getComponent(2).residual.setReference(LF_refs[j])
        problem.stages[j].cost.getComponent(3).residual.setReference(RF_refs[j])

    if problem.stages[0].dynamics.differential_dynamics.constraint_models.__len__() == 1:
        # Left contact
        if problem.stages[0].dynamics.differential_dynamics.constraint_models[0].name == 'left_sole_link':
            force_left.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear)
            force_right.append(np.zeros(3))
            torque_left.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.angular)
            torque_right.append(np.zeros(3))
        # Right contact
        else:
            force_right.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear)
            force_left.append(np.zeros(3))
            torque_right.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.angular)
            torque_left.append(np.zeros(3))
    # Double contact
    elif problem.stages[0].dynamics.differential_dynamics.constraint_models.__len__() == 2:
        force_left.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear)
        force_right.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[1].contact_force.linear)
        torque_left.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[0].contact_force.angular)
        torque_right.append(solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.constraint_datas[1].contact_force.angular)
    else:
        force_right.append(np.zeros(3))
        force_left.append(np.zeros(3))
        torque_left.append(np.zeros(3))
        torque_right.append(np.zeros(3))
    
    print("\nForce L: "+str(round(force_left[t][2]/f_full,2))+" Force R:"+str(round(force_right[t][2]/f_full,2)))

    LF_measured.append(rdata.oMf[LF_id].copy())
    RF_measured.append(rdata.oMf[RF_id].copy())
    LF_references.append(LF_refs[0].copy())
    RF_references.append(RF_refs[0].copy())
    com_measured.append(pin.centerOfMass(rmodel, rdata, x_measured[:nq]))
    pin.computeCentroidalMomentum(rmodel,rdata, x_measured[:nq], x_measured[nq:])
    L_measured.append(rdata.hg.angular.copy())

    device.moveMarkers(LF_refs[0].translation, RF_refs[0].translation)
    print("diocane1")
    #problem.replaceStageCircular(stages_full[t])
    #solver.cycleProblem(problem,stages_full_data[t])

    com_final = com0.copy()
    com_final[:2] = (LF_refs[-1].translation[:2] + RF_refs[-1].translation[:2]) / 2
    com_final[2] = (LF_refs[-1].translation[2] + RF_refs[-1].translation[2]) / 2 + 0.87
    com_cstr = aligator.CentroidalCoMResidual(space.ndx, nu, com_final)
    term_constraint_com = aligator.StageConstraint(
        com_cstr, constraints.EqualityConstraintSet()
    )
    
    #problem.term_cost.components[2][0].residual.setReference(LF_refs[-1])
    #problem.term_cost.components[3][0].residual.setReference(RF_refs[-1])

    # Loop at 1 kHz
    for j in range(Nsimu):
        lowlevel_time += 0.001
        q_current, v_current = device.measureState()
        
        x_measured = shapeState(q_current, 
                                v_current, 
                                nq, 
                                nq + nv, 
                                controlled_ids)  
        pin.forwardKinematics(rmodel, rdata, x_measured[:nq])
        pin.updateFramePlacements(rmodel, rdata)

        current_torque = us[0] - solver.results.controlFeedbacks()[0] @ space.difference(x_measured, xs[0])
        device.execute(current_torque)
        """ if t >= 160 and t < 171:
            print("Force applied")
            device.apply_force(f_disturbance, [0, 0, 0]) """

        u_multibody.append(current_torque)
        x_multibody.append(x_measured)
    
    lowlevel_time = 0
    xs = xs[1:] + [xs[-1]]
    us = us[1:] + [us[-1]]
    xs[0] = x_measured_prev

    problem.x0_init = x_measured_prev
    problem.replaceStageCircular(stages_full[t])
    solver.cycleProblem(problem,stages_full_data[t])
    start = time.time()
    #solver.setup(problem)
    solver.run(problem, xs, us)
    end = time.time()
    time_computation = end - start
    solve_time.append(end - start)
    print("solver.run = " + str(end - start))

    x_measured_prev = x_measured.copy()

    xs = solver.results.xs.tolist().copy()
    us = solver.results.us.tolist().copy()
    K_feedback = solver.results.controlFeedbacks()[0]

#print("Elapsed time:")
#print(np.mean(np.array(solve_time)))


force_left = np.array(force_left)
force_right = np.array(force_right)
torque_left = np.array(torque_left)
torque_right = np.array(torque_right)
solve_time = np.array(solve_time)
LF_measured = np.array(LF_measured)
RF_measured = np.array(RF_measured)
LF_references = np.array(LF_references)
RF_references = np.array(RF_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

""" save_trajectory(x_multibody, u_multibody, com_measured, force_left, force_right, torque_left, torque_right, solve_time, 
                LF_measured, RF_measured, LF_references, RF_references, L_measured, "fulldynamics") """
struct InvertedPendulum
    MAX_v::Float64
    torques::Vector{Float64}
    mass::Float64
    len::Float64
    g::Float64
    mu::Float64
    dt::Float64
end


function InvertedPendulumStep(
        state::Vector{Float64}, # [\theta, \theta_dot] \theta \in [-pi, pi] \theta_dot \in [-MAX_v, MAX_v]
        actionIndx::Int64, # index of torque in Torques=[ ... ]
        Params::InvertedPendulum,#
    )
    MAX_v = Params.MAX_v
    torques = Params.torques
    m = Params.mass
    len = Params.len
    g = Params.g
    mu = Params.mu
    dt = Params.dt

    theta = state[1]
    theta_dot = state[2]
    
    u = torques[min(actionIndx, length(torques))]

    theta_dot_dot = (-mu*theta_dot + m*g*len*sin(theta) + u)/(m*len^2)

    theta_dot_nxt = theta_dot + theta_dot_dot*dt
    if(theta_dot_nxt > MAX_v)
        theta_dot_nxt = MAX_v
    elseif(theta_dot_nxt < -MAX_v)
        theta_dot_nxt = -MAX_v
    end

    theta_nxt = theta + theta_dot_nxt*dt
    if(theta_nxt > pi)
        theta_nxt = theta_nxt - 2*pi
    elseif(theta_nxt < -pi)
        theta_nxt = theta_nxt + 2*pi
    end

    loss = 1.
    absorb = 0

    signal = abs(cos(theta_nxt)-1)
    if(signal < 0.1)
        loss = 0.
        absorb = 1
    end

    next_state = [theta_nxt, theta_dot_nxt]
    return next_state, loss, absorb
end



# EnvParams
MAX_v = 4.
torques = [- 5.; -3; 0.; 3.; 5.]
mass = 1.
len = 1.
g = 9.8
mu = 0.01
dt = 0.05
EnvParams = InvertedPendulum(MAX_v,#
                             torques,#
                             mass,#
                             len,#
                             g,#
                             mu,#
                             dt,#
                            )

function StateAction(
        state::Vector{Float64},#
        actionIndx::Int64,#
        Params::InvertedPendulum,#
    )
    torques = Params.torques
    
    #u = torques[actionIndx]/maximum(abs.(torques))

    #StAcVec = [state; u]

    Na = length(torques)
    Id = Matrix(1.0I, Na, Na)
    Aux = kronecker(Id, state)

    StAcVec = Aux[:, actionIndx]
    return StAcVec
end


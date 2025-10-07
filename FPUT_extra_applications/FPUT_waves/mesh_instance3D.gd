extends MeshInstance3D

# Parametri FPUT
@export var N := 128                # Numero punti catena
@export var alpha_non_linear : float
@export var beta_non_linear : float
@export var dt : float            # Passo temporale
@export var elastic_constant : float
@export var impulse_strength : float
@export var initial_impulse : float

var alpha := 0.0
var beta := 0.0

var x = []
var v = []

func _ready():
	x.resize(N)
	v.resize(N)
	for i in range(N):
		x[i] = 0.0
		v[i] = 0.0
	x[N/2] = initial_impulse #1.0 # impulso iniziale

func _process(delta):
	_integrate()
	
	var mat = get_active_material(0)
	if mat:
		mat.set_shader_parameter("wave_positions", x)
		mat.set_shader_parameter("wave_speeds", v)

func _input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_SPACE:
			# Toggle modello lineare / non lineare
			if alpha == 0.0 and beta == 0.0:
				alpha = alpha_non_linear
				beta = beta_non_linear
				print("Modalità: NON lineare")
			else:
				alpha = 0.0
				beta = 0.0
				print("Modalità: lineare")
	
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		# Click → impulso in base alla posizione X
		var ray = get_viewport().get_camera_3d().project_ray_origin(event.position)
		var dir = get_viewport().get_camera_3d().project_ray_normal(event.position)
		var from = ray
		var to = ray + dir * 100.0
		var space_state = get_world_3d().direct_space_state
		var query = PhysicsRayQueryParameters3D.create(from, to)
		var result = space_state.intersect_ray(query)
		if result and result.has("position"):
			var hit_x = result.position.x
			var idx = int((hit_x / 10.0 + 0.5) * float(N))
			idx = clamp(idx, 0, N - 1)
			v[idx] += impulse_strength

func _integrate():
	var a = []
	a.resize(N)
	
	for i in range(1, N-1):
		var dx1 = x[i+1] - x[i]
		var dx2 = x[i] - x[i-1]
		a[i] = elastic_constant * (dx1 - dx2) \
			 + alpha * (pow(dx1,2) - pow(dx2,2)) \
			 + beta  * (pow(dx1,3) - pow(dx2,3))
	
	for i in range(1, N-1):
		v[i] += a[i] * dt
		x[i] += v[i] * dt

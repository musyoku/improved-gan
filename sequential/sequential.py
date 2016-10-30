import copy, json, types
import chainer
import link
import function

def get_weight_initializer(initializer, init_std):
	if initializer.lower() == "normal":
		return chainer.initializers.Normal(init_std)
	if initializer.lower() == "glorotnormal":
		return chainer.initializers.GlorotNormal(init_std)
	if initializer.lower() == "henormal":
		return chainer.initializers.HeNormal(init_std)
	raise Exception()

class Sequential(object):
	def __init__(self, weight_initializer="Normal", weight_init_std=1):
		self._layers = []
		self.links = []

		self.weight_initializer = weight_initializer	# Normal / GlorotNormal / HeNormal
		self.weight_init_std = weight_init_std

	def add(self, layer):
		if isinstance(layer, link.Link) or isinstance(layer, function.Function):
			self._layers.append(layer)
		elif isinstance(layer, function.Activation):
			self._layers.append(layer.to_function())
		else:
			raise Exception()

	def layer_from_dict(self, dict):
		if "_link" in dict:
			if hasattr(link, dict["_link"]):
				args = self.dict_to_layer_init_args(dict)
				return getattr(link, dict["_link"])(**args)
		if "_function" in dict:
			if hasattr(function, dict["_function"]):
				args = self.dict_to_layer_init_args(dict)
				return getattr(function, dict["_function"])(**args)
		raise Exception()

	def dict_to_layer_init_args(self, dict):
		args = copy.deepcopy(dict)
		remove_keys = []
		for key, value in args.iteritems():
			if key[0] == "_":
				remove_keys.append(key)
		for key in remove_keys:
			del args[key]
		return args

	def get_weight_initializer(self):
		return get_weight_initializer(self.weight_initializer.lower(), self.weight_init_std)

	def layer_to_chainer_link(self, layer):
		if hasattr(layer, "_link"):
			if layer.has_multiple_weights() == True:
				if isinstance(layer, link.GRU):
					layer._init = self.get_weight_initializer()
					layer._inner_init = self.get_weight_initializer()
				elif isinstance(layer, link.LSTM):
					layer._lateral_init  = self.get_weight_initializer()
					layer._upward_init  = self.get_weight_initializer()
					layer._bias_init = self.get_weight_initializer()
					layer._forget_bias_init = self.get_weight_initializer()
				elif isinstance(layer, link.StatelessLSTM):
					layer._lateral_init  = self.get_weight_initializer()
					layer._upward_init  = self.get_weight_initializer()
				elif isinstance(layer, link.StatefulGRU):
					layer._init = self.get_weight_initializer()
					layer._inner_init = self.get_weight_initializer()
			else:
				layer._initialW = self.get_weight_initializer()
			return layer.to_link()
		if hasattr(layer, "_function"):
			return layer
		raise Exception()

	def build(self):
		json = self.to_json()
		self.from_json(json)

	def to_dict(self):
		layers = []
		for layer in self._layers:
			config = layer.to_dict()
			dic = {}
			for key, value in config.iteritems():
				if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
					dic[key] = value
			layers.append(dic)
		return {
			"layers": layers,
			"weight_initializer": self.weight_initializer,
			"weight_init_std": self.weight_init_std
		}

	def to_json(self):
		result = self.to_dict()
		return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

	def from_json(self, str):
		self.links = []
		self._layers = []
		attributes = {}
		dict_array = json.loads(str)
		self.from_dict(dict_array)

	def from_dict(self, dict):
		self.weight_initializer = dict["weight_initializer"]
		self.weight_init_std = dict["weight_init_std"]
		for i, layer_dict in enumerate(dict["layers"]):
			layer = self.layer_from_dict(layer_dict)
			link = self.layer_to_chainer_link(layer)
			self.links.append(link)
			self._layers.append(layer)

	def __call__(self, x, test=False):
		for i, link in enumerate(self.links):
			if isinstance(link, function.dropout):
				x = link(x, train=not test)
			elif isinstance(link, chainer.links.BatchNormalization):
				x = link(x, test=test)
			else:
				x = link(x)
		return x

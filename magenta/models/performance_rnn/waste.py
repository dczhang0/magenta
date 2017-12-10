# class PRNN(object):
#
#     def __init__(self, model, session):
#
#         self.session = session
#         self.model = model
#         self.initialise()
#
#     def initialise(self):
#         self.state, self.pmf = self.session.run([])
#
#     def get_pmf(self):
#         return self.pmf
#
#     def get_cdf(self):
#         return np.cumsum(self.pmf)
#
#     def add_action(self, action):
#         result = self.session.run([self.model.state, self.model.pmf], feed_dict=dict(self.model.state=self.state, self.model.x=action))
#         self.state, self.pmf = result
#
#     @classmethod
#     def from_checkpoint(cls, checkpoint_filename):
#         # stuff
#         return PRNN(model, session)
#
#     @classmethod
#     def test_primer_sequence(cls, primer, prnn):
#         prnn.initialise()
#         for action in primer:
#             prnn.add_action(action)
#         libo_pmf = prnn.get_pmf()
#
#         tf_pmf = tf_function(primer)
#
#         assert np.allclose(libo_pmf, tf_pmf)
#
#
# class Constraints(object):
#
#     def __init__(self, t, p, e, b):
#         self.t = t
#         self.p = p
#         self.e = e
#         self.b = b
#
#     @classmethod
#     def from_file(cls, filename):
#         return Constraints(t, p, e, b)
#
#     @classmethod
#     def primer_from_midi_file(cls, filename, t_end):
#         return Constraints(t, p, e, b)
#
#     def valid(self, sequence):
#         return True or False
#
#
# class Sampler(object):
#
#     def __init__(self, prnn, constraints):
#         self.prnn = prnn
#         self.constraints = constraints
#
#     def sample(self):
#         while True:
#             raise
#
#
# if __name__ == '__main__':
#
#     prnn = PRNN.from_checkpoint(checkpoint_filename)
#     primer = something
#     PRNN.test_primer_sequence(primer, prnn)
#
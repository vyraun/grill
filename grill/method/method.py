class Method(object):
    def __init__(self, name):
        self.name = name

    def register_with_engine(self, engine):
        engine.register_callback('pre-step', self.name + ' pre-step',
                lambda engine, episode: self.pre_step(engine, episode))
        engine.register_callback('post-step', self.name + ' post-step',
                lambda engine, episode: self.post_step(engine, episode))
        engine.register_callback('pre-episode', self.name + ' pre-episode',
                lambda engine, episode: self.pre_episode(engine, episode))
        engine.register_callback('post-episode', self.name + ' post-episode',
                lambda engine, episode: self.post_episode(engine, episode))

    def pre_step(self, engine, episode): pass
    def post_step(self, engine, episode): pass
    def pre_episode(self, engine, episode): pass
    def post_episode(self, engine, episode): pass

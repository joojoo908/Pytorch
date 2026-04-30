from majestro_navmesh_env import MajestroNavMeshEnv


Env = MajestroNavMeshEnv


def make_env(**kwargs):
    return MajestroNavMeshEnv(**kwargs)

__all__ = ["PlatoonEnv"]


def __getattr__(name: str):
	if name == "PlatoonEnv":
		from .platoon_env import PlatoonEnv

		return PlatoonEnv
	raise AttributeError(name)

import flaskr.Tesi.components as components


class modelFactory:
    free_models: "list[components.models.Model]" = []
    models_in_use: "list[components.models.Model]" = []
    currently_in_use: int = 0
    max_size: int = 5

    def __init__(self) -> None:
        for _ in range(self.max_size):
            model = components.model_factory(
                multiple=True,
                gpu_enable=True,
                model_path="./flaskr/Tesi/mpi/pose_iter_160000.caffemodel",
                proto_path="./flaskr/Tesi/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
            )
            model.init_net()
            self.free_models.append(model)

    def aquire_model(self):
        if self.currently_in_use == self.max_size:
            return None
        else:
            self.currently_in_use += 1
            model = self.free_models.pop()
            self.models_in_use.append(model)
            return model

    def release_model(self, model: components.models.Model):
        if self.currently_in_use == 0:
            raise Exception
        else:
            self.models_in_use.remove(model)
            self.free_models.append(model)
            self.currently_in_use -= 1


class painterFactory:
    free_painters: "list[components.painters.Painter]" = []
    painters_in_use: "list[components.painters.Painter]" = []
    currently_in_use: int = 0
    max_size: int = 5

    def __init__(self) -> None:
        for _ in range(self.max_size):
            painter = components.painter_factory(private=True)
            self.free_painters.append(painter)

    def aquire_painter(self):
        if self.currently_in_use == self.max_size:
            return None
        else:
            self.currently_in_use += 1
            painter = self.free_painters.pop()
            self.painters_in_use.append(painter)
            return painter

    def release_painter(self, model: components.painters.Painter):
        if self.currently_in_use == 0:
            raise Exception
        else:
            self.painters_in_use.remove(model)
            self.free_painters.append(model)
            self.currently_in_use -= 1

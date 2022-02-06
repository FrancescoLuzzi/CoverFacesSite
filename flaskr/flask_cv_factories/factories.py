import flaskr.Tesi.components as components


class modelFactory:
    __free_models: "list[components.models.Model]" = []
    __models_in_use: "list[components.models.Model]" = []
    __currently_in_use: int = 0
    __max_size: int = 5

    def __init__(self) -> None:
        for _ in range(self.__max_size):
            model = components.model_factory(
                multiple=True,
                gpu_enable=True,
                model_path="./flaskr/Tesi/mpi/pose_iter_160000.caffemodel",
                proto_path="./flaskr/Tesi/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
            )
            model.init_net()
            self.__free_models.append(model)

    def aquire_model(self):
        if self.__currently_in_use == self.__max_size:
            return None
        else:
            self.__currently_in_use += 1
            model = self.__free_models.pop()
            self.__models_in_use.append(model)
            return model

    def release_model(self, model: components.models.Model):
        if self.__currently_in_use == 0:
            raise Exception
        else:
            self.__models_in_use.remove(model)
            self.__free_models.append(model)
            self.__currently_in_use -= 1


class painterFactory:
    __free_painters: "list[components.painters.Painter]" = []
    __painters_in_use: "list[components.painters.Painter]" = []
    __currently_in_use: int = 0
    __max_size: int = 5

    def __init__(self) -> None:
        for _ in range(self.__max_size):
            painter = components.painter_factory(private=True)
            self.__free_painters.append(painter)

    def aquire_painter(self):
        if self.__currently_in_use == self.__max_size:
            return None
        else:
            self.__currently_in_use += 1
            painter = self.__free_painters.pop()
            self.__painters_in_use.append(painter)
            return painter

    def release_painter(self, model: components.painters.Painter):
        if self.__currently_in_use == 0:
            raise Exception
        else:
            self.__painters_in_use.remove(model)
            self.__free_painters.append(model)
            self.__currently_in_use -= 1

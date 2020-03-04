from functools import partial
def define_model_fit(**model_fit_args):
    """
    @ Define_model_fit is a decorator that defines model.fit() and passes 
      arguements when the model is built with a function
    @ To be used in a training pipeline when model.fit() has to be called
      and different callbacks have to be added dynamically
    ** Inputs:
      key-value pair of arguments to pass into model.fit()
    """
    def visualise_wrapper(model_builder):
        def model(*args, **kwargs):
            model_ = model_builder(*args, **kwargs)
            old_fit = model_.fit
            new_fit = partial(old_fit,**model_fit_args)
            model_.fit = new_fit
            return model_
        return model
    return visualise_wrapper


if __name__ == "__main__":
  pass
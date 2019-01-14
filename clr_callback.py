from keras.callbacks import *


class CyclicLR(Callback):

    def __init__(self, 
                 batches_per_epoch,
                 epoch_per_cycle = 2,
                 smoothing_factor = 5,
                 
                 lr_min = -5.0,
                 lr_max = 0.0,                 
                 lr_depth = 10,                 
                 lr_decay = 0.95,

                 momentum_depth = 1.25,
                 
                 decay_hyper_params = True
                ):
        super(CyclicLR, self).__init__()
        
        self.batches_per_epoch = batches_per_epoch
        self.epoch_per_cycle = epoch_per_cycle
        self.smoothing_factor = smoothing_factor
        
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_depth = lr_depth
        self.lr_decay = lr_decay
       
        self.momentum_depth = momentum_depth
                
        self.decay_hyper_params = decay_hyper_params

        
    def _reset(self):
           
        self.base_lr = 0.0
        self.max_lr = 1.0
        
        self.base_clipnorm = 1.0

        self.base_momentum = 1.0
        self.min_momentum = 0.50
        
        self.base_decay = 0.0

        self.rate_schedule = []
        self.clipnorm_schedule = []
        self.momentum_schedule = []
        self.decay_schedule = []

        self.ascending = False

        self.loss_history = []
        self.epoch_loss_history = []
        self.lr_history = []        
        self.clipnorm_history = []
        self.momentum_history = []
        self.decay_history = []

        # redo finding the optimum values
        self.find_all()
     
  
    def find_all(self):
      
        self.searching = True   
        self.searching_lr = True
        self.searching_clip = True
        self.searching_momentum = True
        self.searching_decay = True
        
        self.find_lr()
        
        
    def find_lr(self):
        # create an x value for each batch
        xes = np.linspace(self.lr_min, self.lr_max, self.batches_per_epoch)
             
        # increases the learning rate exponentially as we search for the best value
        self.rate_schedule = []
        for i in range(self.batches_per_epoch):
            self.rate_schedule.append( math.exp( xes[i] ) )
        
        # set defaults for searching
        K.set_value(self.model.optimizer.lr, np.float32(self.rate_schedule[0])) 
        self.model.optimizer.clipnorm = np.float32(1.0)
        K.set_value(self.model.optimizer.momentum, np.float32(0.50))
        K.set_value(self.model.optimizer.decay, np.float32(0.0))        
           
        print("find LR: ", len(self.rate_schedule))

  
    def find_clipnorm(self):      
        # create an x value for each batch
        xes = np.linspace(0.0, 1.0, self.batches_per_epoch)
        
        # increase the clipnorm at a rate of x^e between 0.5 and 1.5
        self.clipnorm_schedule = []
        for i in range(self.batches_per_epoch):
            self.clipnorm_schedule.append( 10.0 * ( xes[i] ** math.e ) ) 
          
        # set defaults for searching 
        K.set_value(self.model.optimizer.lr, np.float32(self.max_lr)) # use the largest LR
        self.model.optimizer.clipnorm = np.float32(self.clipnorm_schedule[0])
        K.set_value(self.model.optimizer.momentum, np.float32(0.50))
        K.set_value(self.model.optimizer.decay, np.float32(0.0))
        print("find clip: ", len(self.clipnorm_schedule))
  
   
    def find_momentum(self):      
        # create an x value for each batch
        xes = np.linspace(0.0, 1.0, self.batches_per_epoch)
        
        # increase the momentum at a rate of x^e between 0.0 and 1.0
        self.momentum_schedule = []
        for i in range(self.batches_per_epoch):
            self.momentum_schedule.append( ( xes[i] ** math.e ) ) 
        
        # set defaults for searching 
        K.set_value(self.model.optimizer.lr, np.float32(self.base_lr)) # use the min LR
        self.model.optimizer.clipnorm = np.float32(self.base_clipnorm)
        K.set_value(self.model.optimizer.momentum, np.float32(self.momentum_schedule[0]))
        K.set_value(self.model.optimizer.decay, np.float32(0.0))    
        print("find momentum: ", len(self.momentum_schedule))
        
        
    def find_decay(self):      
        # create an x value for each batch
        xes = np.linspace(0.0, 1.0, self.batches_per_epoch)
        
        # increase the momentum at a rate of x^e between 0.0 and 1.0
        self.decay_schedule = []
        for i in range(self.batches_per_epoch):
            self.decay_schedule.append( ( xes[i] ** math.e ) ) 
        
        # set defaults for searching 
        K.set_value(self.model.optimizer.lr, np.float32(self.base_lr)) # use the min LR
        self.model.optimizer.clipnorm = np.float32(self.base_clipnorm)
        K.set_value(self.model.optimizer.momentum, np.float32(self.base_momentum))
        K.set_value(self.model.optimizer.decay, np.float32(self.decay_schedule[0]))    
        print("find decay: ", len(self.decay_schedule))

        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.iteration = 0

        assert hasattr(self.model.optimizer, 'momentum'), \
            'Optimizer must have a "momentum" attribute.'
        
        self._reset()
        
                
    def on_epoch_end( self , epoch, logs = {}):
            
      logs = logs or {}
      loss = np.float32(logs.get('loss'))
      self.iteration = 0

      # if we are in the first few epochs where we're searching for the best values
      if self.searching:
        
        if(self.searching_lr):
            
            # we've finished searching LR, compile the results and move on
            self.searching_lr = False

            # smooth the data, each batch can have a lot of noise, get the lowest loss once smoothed and use that value
            smoothed = scipy.signal.medfilt(self.epoch_loss_history, self.smoothing_factor)
            index_of_best = np.argmin(smoothed)        
            best_lr = self.rate_schedule[index_of_best]


            # search is done, set our top and bottom values
            self.base_lr = best_lr / self.lr_depth
            self.max_lr = best_lr

            # set the LR for the test
            K.set_value(self.model.optimizer.lr, np.float32(self.base_lr))

            # now find the clipnorm
            self.find_clipnorm()


        elif(self.searching_clip):
            # we've finished searching clipnorm, compile the results and move on
            self.searching_clip = False            
            
            # smooth the data, each batch can have a lot of noise, get the lowest loss once smoothed and use that value
            smoothed = scipy.signal.medfilt(self.epoch_loss_history, self.smoothing_factor)
            index_of_best = np.argmin(smoothed)        
            self.base_clipnorm = self.clipnorm_schedule[index_of_best]

            self.model.optimizer.clipnorm = np.float32(self.base_clipnorm)
            
            # now find the momentum
            self.find_momentum()
          
        elif(self.searching_momentum):
            # we've finished searching momentum, compile the results and move on
            self.searching_momentum = False            
            
            # smooth the data, each batch can have a lot of noise, get the lowest loss once smoothed and use that value
            smoothed = scipy.signal.medfilt(self.epoch_loss_history, self.smoothing_factor)
            index_of_best = np.argmin(smoothed)
            
            self.base_momentum = self.momentum_schedule[index_of_best]
            self.min_momentum = self.base_momentum / 2.0
            
            K.set_value(self.model.optimizer.momentum, np.float32(self.base_momentum))
            
            # now find the momentum
            self.find_decay()    
            
        elif(self.searching_decay):
          
            # we've finished searching decay, compile the results and move on
            self.searching_decay = False            
            
            # smooth the data, each batch can have a lot of noise, get the lowest loss once smoothed and use that value
            smoothed = scipy.signal.medfilt(self.epoch_loss_history, self.smoothing_factor)
            index_of_best = np.argmin(smoothed)
            
            self.base_decay = self.decay_schedule[index_of_best]
            
            K.set_value(self.model.optimizer.decay, np.float32(self.base_decay))
            
            # we found all the values, print them out and stop searching
            
            self.searching = False
            print( "Found the following values: ")          
            print( "base_lr: ", self.base_lr )
            print( "max_lr: ", self.max_lr )
            print( "base_clipnorm: ", self.base_clipnorm )
            print( "base_momentum: ", self.base_momentum )
            print( "min_momentum: ", self.min_momentum )
            print( "base_decay: ", self.base_decay )
        
      if not self.searching:
        # Flip the value of acending/decending every epoch
        self.ascending = not self.ascending
        
        # cyles are Learning Rate Acending and Learning Rate Decending
        if self.ascending:
          old_clipnorm = self.base_clipnorm 
          old_decay = self.base_decay           
          # use hyper Param decay if enabled
          if(self.decay_hyper_params):         
              self.base_lr *= self.lr_decay
              self.base_clipnorm *= self.lr_decay
              self.base_momentum = ((self.base_momentum - self.min_momentum) * self.lr_decay) + self.min_momentum
              self.base_decay *= self.lr_decay
                        
          # set the parameter schedules                        
          self.rate_schedule = np.linspace(self.base_lr, self.max_lr, self.batches_per_epoch)
          self.clipnorm_schedule = np.linspace(old_clipnorm, self.base_clipnorm, self.batches_per_epoch)
          self.momentum_schedule = np.linspace(self.base_momentum, self.min_momentum, self.batches_per_epoch)
          self.decay_schedule = np.linspace(old_decay, self.base_decay, self.batches_per_epoch)
  
        else:
          old_clipnorm = self.base_clipnorm 
          old_decay = self.base_decay           
          # use hyper Param decay if enabled
          if(self.decay_hyper_params):                        
              self.max_lr = ((self.max_lr - self.base_lr) * self.lr_decay) + self.base_lr
              self.base_clipnorm *= self.lr_decay
              self.min_momentum *= self.lr_decay
              self.base_decay *= self.lr_decay
          
          # set the parameter schedules
          self.rate_schedule = np.linspace(self.max_lr, self.base_lr, self.batches_per_epoch)
          self.clipnorm_schedule = np.linspace(old_clipnorm, self.base_clipnorm, self.batches_per_epoch)
          self.momentum_schedule = np.linspace(self.min_momentum, self.base_momentum, self.batches_per_epoch)
          self.decay_schedule = np.linspace(old_decay, self.base_decay, self.batches_per_epoch)

      
      # clear the loss history for this epoch
      self.epoch_loss_history = []
      
      
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        
        loss = np.float32(logs.get('loss'))

        # keep metrics for the history, is is mostly for debugging
        self.loss_history.append(loss)
        self.epoch_loss_history.append(loss)
        self.lr_history.append(K.get_value(self.model.optimizer.lr))
        self.clipnorm_history.append(self.model.optimizer.clipnorm)#K.get_value(self.model.optimizer.clipnorm))
        self.momentum_history.append(K.get_value(self.model.optimizer.momentum))
        self.decay_history.append(K.get_value(self.model.optimizer.decay))
        
        # set the values for this point in the cycle
        
        if self.searching:
            # if we are seraching then we keep all the parameters except one constant
            if(self.searching_lr):
                K.set_value(self.model.optimizer.lr, np.float32(self.rate_schedule[self.iteration]))
 
            elif(self.searching_clip):
                self.model.optimizer.clipnorm = np.float32(self.clipnorm_schedule[self.iteration])
          
            elif(self.searching_momentum):
                K.set_value(self.model.optimizer.momentum, np.float32(self.momentum_schedule[self.iteration]))
          
            elif(self.searching_decay):     
                K.set_value(self.model.optimizer.decay, np.float32(self.decay_schedule[self.iteration]))
                            
        else:
            # if we are running, then we adjust all parameters acording to their scales
            K.set_value(self.model.optimizer.lr, np.float32(self.rate_schedule[self.iteration]))
            self.model.optimizer.clipnorm = np.float32(self.clipnorm_schedule[self.iteration])
            K.set_value(self.model.optimizer.momentum, np.float32(self.momentum_schedule[self.iteration]))
            K.set_value(self.model.optimizer.decay, np.float32(self.decay_schedule[self.iteration]))

                            
                            
        self.iteration += 1
        
        

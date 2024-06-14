import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from models.model_base import PreTrainedModelWrapper
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Conversation,
    TrainingArguments,
    pipeline,
)
from trl import DPOTrainer


class AutoDPOModelForCausalLM(PreTrainedModelWrapper):
    """
    An autoregressive model with support for custom modules in addition to the language model.
    This class inherits from `PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the custom module class you designed. Currently, the supported args are: ______
    """

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ("reference_model", "use_system_msg", "quantized")
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] =  "auto"
        self.is_ref_model = kwargs.get("reference_model", True)
        self.use_sys_msg = kwargs.get("use_system_msg", False)
        self.quantized = kwargs.get("quantized", False)
        super().__init__(pretrained_model, **kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        
        if kwargs.get("quantized", False):
            print("Quantized!")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        
            kwargs["quantization_config"] = bnb_config
        
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        ## TODO: Is there a better way of integrating the DPOTrainer functions into the model?
        
        model.args = TrainingArguments(
            output_dir="llama3_new",               # directory to save and repository id
            num_train_epochs=1,                     # number of training epochs
            per_device_train_batch_size=1,         # batch size per device during training
            per_device_eval_batch_size=1,           # batch size for evaluation
            gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
            gradient_checkpointing=True,            # use gradient checkpointing to save memory
            optim="adamw_torch_fused",              # use fused adamw optimizer
            learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
            max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
            lr_scheduler_type="cosine",             # use cosine learning rate scheduler
            logging_steps=25,                       # log every 25 steps
            save_steps=500,                         # when to save checkpoint
            save_total_limit=2,                     # limit the total amount of checkpoints
            evaluation_strategy="steps",            # evaluate every 1000 steps
            eval_steps=700,                         # when to evaluate
            bf16=True,                              # use bfloat16 precision
            tf32=True,                              # use tf32 precision
            push_to_hub=False,                      # push model to hub
            report_to="tensorboard",                # report metrics to tensorboard
        )
        
        model.dpo_args = {
            "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
            "loss_type": "sigmoid"                  # The loss type for DPO.
        }

        model.prompt_length = 402
        model.max_seq_length = 912
        
        model.dpo_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model.dpo_tokenizer.pad_token = model.dpo_tokenizer.eos_token
        model.dpo_tokenizer.padding_side = 'left' # to prevent errors with FA
        model.dpo_tokenizer.truncation_side = 'left' # to prevent cutting off last generation
        
        model.dpo_trainer = DPOTrainer(
            model.pretrained_model,
            ref_model=None,
            args=model.args,
            train_dataset=Dataset.from_dict({}),
            eval_dataset=Dataset.from_dict({}),
            tokenizer=model.dpo_tokenizer,
            max_length=model.max_seq_length,
            max_prompt_length=model.prompt_length,
            beta=model.dpo_args["beta"],
            loss_type=model.dpo_args["loss_type"],
        )
        
        print(model.pretrained_model.get_memory_footprint())
        
        return model

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            output_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        # raise NotImplementedError
        ###############################################################

        # As this is primarily adapted from DPOTrainer functions, the forward pass isn't really used

        outputs = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )

        return outputs

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        # raise NotImplementedError
        ###############################################################
        
        # TODO: this is a horrible mess that needs to be cleaned up
        
        system_msg = {"role": "system", "content": "You are an expert professor, teaching a student how to solve a problem by providing a full explanation of the solution."}
        
        chosen_list = []
        rejected_list = []
        
        for idx in tqdm(range(len(batch["prompt"]))):
            msg_qn = {"role": "user", "content": batch["prompt"][idx]}
            msg_chosen = {"role": "assistant", "content": batch["chosen"][idx]}
            msg_rejected = {"role": "assistant", "content": batch["rejected"][idx]}
            
            dpo_dataset_dict = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
            }
            
            dpo_dataset_dict["prompt"] = [system_msg, msg_qn]
            dpo_dataset_dict["chosen"] = [msg_chosen]
            dpo_dataset_dict["rejected"] = [msg_rejected]
        
            # dataset = Dataset.from_dict(dpo_dataset_dict)
            
            def process(row):
                row["prompt"] = self.dpo_trainer.tokenizer.apply_chat_template(row["prompt"], tokenize=False)
                row["chosen"] = self.dpo_trainer.tokenizer.apply_chat_template(row["chosen"], tokenize=False)
                row["rejected"] = self.dpo_trainer.tokenizer.apply_chat_template(row["rejected"], tokenize=False)
                return row
            
            dpo_dataset_dict["prompt"] = self.dpo_trainer.tokenizer.apply_chat_template(dpo_dataset_dict["prompt"], tokenize=False)
            dpo_dataset_dict["chosen"] = self.dpo_trainer.tokenizer.apply_chat_template(dpo_dataset_dict["chosen"], tokenize=False)
            dpo_dataset_dict["rejected"] = self.dpo_trainer.tokenizer.apply_chat_template(dpo_dataset_dict["rejected"], tokenize=False)
            
            # .map() is breaking for some reason
            # dataset = dataset.map(
            #     process,
            #     num_proc=1,
            #     load_from_cache_file=False,
            # )
        
            data = self.dpo_trainer.tokenize_row(dpo_dataset_dict)
            
            new_ds_dict = {}
            for k,v in data.items():
                new_ds_dict[k] = [v]
        
            dataset = Dataset.from_dict(new_ds_dict)
            
            ds_loader = self.dpo_trainer.get_eval_dataloader(dataset)

            random_batch_dataset = ds_loader.dataset.select([0])
            random_batch = self.dpo_trainer.data_collator(random_batch_dataset)
            random_batch = self.dpo_trainer._prepare_inputs(random_batch)
            
            with torch.no_grad():
                if self.is_ref_model:
                    # The ref model is just the model without lora
                    with self.dpo_trainer.null_ref_context():
                        (
                            step_chosen_logps,
                            step_rejected_logps,
                            _,
                            _,
                        ) = self.dpo_trainer.concatenated_forward(self.dpo_trainer.model, random_batch)
                else:
                    (
                        step_chosen_logps,
                        step_rejected_logps,
                        _,
                        _,
                    ) = self.dpo_trainer.concatenated_forward(self.dpo_trainer.model, random_batch)
            
            chosen_list.append(step_chosen_logps[0].item())
            rejected_list.append(step_rejected_logps[0].item())

        chosen_logps = torch.tensor(chosen_list)
        rejected_logps = torch.tensor(rejected_list)

        return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        # output_dict = {
        #     "chosen_rewards": [],
        #     "rejected_rewards": []
        # }

        ########################################################################
        # TODO: Please implement the prediction step that computes the rewards
        # ======================================================================
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        # raise NotImplementedError
        ########################################################################

        # Rewards come directly from the DPOTrainer loss

        losses, chosen_rewards, rejected_rewards = self.dpo_trainer.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        output_dict = {
            "chosen_rewards": chosen_rewards.tolist(),
            "rejected_rewards": rejected_rewards.tolist()
        }

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        
        # Here we'll use a conversational pipeline to generate the MCQA answers
        # Allows for easy multi-step generation
        
        device = None if self.quantized else self.pretrained_model.device
        
        chatbot = pipeline("conversational", model=self.pretrained_model, tokenizer=tokenizer, device=device)
        
        system_msg = {"role": "system", "content": "You are an expert professor, teaching a student how to solve a problem by providing a full explanation of the solution."}

        mcq_msg = {"role": "user", "content": "Now, based off of the explanation you have given, give the correct answer as a single letter only, e.g. `A`, `B`, `C` or `D`."}

        dataset = Dataset.from_dict(batch)
        
        output_dict = {"preds": []}

        for dp in tqdm(dataset):
            conversation = Conversation()
            if self.use_sys_msg:
                conversation.add_message(system_msg)
            conversation.add_message({"role": "user", "content": dp["question"]})

            conversation = chatbot(
                conversation,
                max_new_tokens=2048,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            conversation.add_message(mcq_msg)
            
            # print("init done!!")
            # print(f"True answer: {dp['answer']}")
            
            # The generation of the clean mcq answer can be inconsistent
            # So, average over 10 tries to get the most common answer
            sample_answers = []
            answers_list = []
            for i in range(10):
                conversation_sample = chatbot(
                    copy.deepcopy(conversation),
                    max_new_tokens=16,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                answer_string = conversation_sample.messages[-1]["content"]
                answers_list.append(answer_string)
                
                # print(answer_string)
                # print("============")
                
                # If the generation starts with a letter, use that
                if answer_string[0] in ["A", "B", "C", "D", "E"]:
                    sample_answers.append(answer_string[0])
                    continue
                
                # or if the generation starts with `<letter>` (also common)
                if answer_string[:3] in ["`A`", "`B`", "`C`", "`D`", "`E`"]:
                    sample_answers.append(answer_string[1])
                    continue
                
                # or finally if the generation contains `<letter>` (can happen sometimes)
                for match in ["`A`", "`B`", "`C`", "`D`", "`E`"]:
                    if match in answer_string:
                        sample_answers.append(match[1])
                        continue
            
            if len(sample_answers) == 0:
                raise ValueError("No valid answer found.")
            
            # Get the most frequent answer
            answer = max(set(sample_answers), key=sample_answers.count)
                
            output_dict["preds"].append(answer)

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        # raise NotImplementedError
        ########################################################################

        return output_dict

class AutoDPOModelForSeq2SeqLM(PreTrainedModelWrapper):
    r"""
    A seq2seq model with support for custom modules in addition to the transformer model.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to any `CustomModule` classes.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            ouput_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        ouput_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return ouput_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the dpo loss function to compute the rewards
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict

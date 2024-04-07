#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import timeit
import gradio as gr


from models.base_model import BaseModel
from utils.enums import QuestionModelName, AnswerModelName, DistractorModelName
from utils.statistics import compare_question_models, compare_answering_model, compare_distractor_models, compare_pipelines, compare_pipeline_results
from utils.trainer import Trainer, MetricType


def main(mode):
    """ Running the pipeline """
    question_model = QuestionModelName.FIONA
    answer_model = AnswerModelName.DEEPSET
    distractor_model = DistractorModelName.BART

    print("=======================================================================================================")
    if mode == "regular":
        print("RUNNING QUESTION GENERATION MODE \n")
        basemodel = BaseModel(question_model, answer_model, distractor_model)
        context = input("\nPlace your context:\n")
        answer = input("\nPlace your answer (empty for no answer):\n")
        q, a, d = basemodel.generate(context, answer=answer, parallel=True)

        print(f"\nQUESTION GENERATED USING: {question_model.value}")
        print(f"QUESTION: {q}\n")
        print(f"ANSWER GENERATED USING: {answer_model.value}")
        print(f"ANSWER: {a}\n")
        print(f"DISTRACTORS GENERATED USING: {distractor_model.value}")
        print(f"DISTRACTORS: {d}\n")

    if mode.startswith("interactive"):
        basemodel = BaseModel(question_model, answer_model, distractor_model)

        def generate(inp, history):
            inputs = inp.split("|")
            if len(inputs) == 1:
                q, a, d = basemodel.generate(inputs[0], parallel=True)
            else:
                q, a, d = basemodel.generate(inputs[0], answer=inputs[1], parallel=True)
            return f"""
            Question: {q}\n
            Answer: {a}\n
            Distractor: {d} 
            """

        gr.ChatInterface(
            generate,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="Provide either a context or both a context and answer separated by |",
                               container=False, scale=8),
            title="Science Quiz Generator",
            description="Provide either a context or both a context and answer separated by |",
            theme="soft",
            retry_btn="Retry",
            undo_btn="Delete Previous",
            clear_btn="Clear",
        ).launch()

    if mode.startswith("train"):
        print("RUNNING MODEL TRAINING MODE\n")
        if mode.split("_")[-1] == "q":
            print("TRAINING QUESTION MODEL WITH")
            print()
            print(f"MODEL: flan-t5-base")
            print()
            print(f"METRIC: ROUGE")
            print()
            trainer = Trainer(QuestionModelName.T5FLAN, MetricType.ROUGE)
            trainer.train_questions('questions-1', with_answer=True)
        else:
            print("TRAINING DISTRACTOR MODEL WITH")
            print()
            print(f"MODEL: flan-t5-base")
            print()
            print(f"METRIC: BLEURT")
            print()
            trainer = Trainer(DistractorModelName.FLAN, MetricType.BLEURT)
            trainer.train_distractors('distractors-2')

    if mode == "evaluate":
        print("RUNNING EVALUATION MODE\n")
        print("COMPARING QUESTION MODELS")
        compare_question_models()
        print("COMPARING ANSWERING MODELS")
        compare_answering_model()
        print("COMPARING DISTRACTOR MODELS")
        compare_distractor_models()
        print("COMPARING PIPELINES")
        compare_pipelines()
        print("COMPARING PIPELINE RESULTS")
        compare_pipeline_results()

    print("=======================================================================================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline with specified models.")
    parser.add_argument("mode", choices=["regular", "interactive", "train_q", "train_d", "evaluate"],
                        help="Choose question model")

    args = parser.parse_args()
    main(args.mode)

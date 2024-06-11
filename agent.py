import numpy as np

from snake import Color, BLOCK_SIZE, Snake
from model import SnakeModel
from _agent import Agent


class Agents:
    def __init__(self,
                 w,
                 h,
                 screen,
                 font,
                 n_agents=10,):
        self.w = w
        self.h = h
        self.agents = dict()
        self.screen = screen
        self.font = font
        self.best_scores = {
            "fitness": 0,
            "score": 0,
        }

        self.this_gen_best_scores = {
            "fitness": 0,
            "score": 0,
            "avg_score": 0,
            "avg_fitness": 0,
        }
        self.gen = 0
        for i in range(n_agents):
            self.agents[i] = {
                "agent": Agent(snake=Snake(w=w, h=h),
                               model=SnakeModel),
                "fitness": 0,
                "color": Color.GREEN.value,
                "steps": 0,
                "gen": self.gen,
                "score": 0,
                "agent_no": i,
            }

        self.__initialize()

    def __initialize(self):
        """
        Setting some initial values for the Agents
        """
        for i in range(len(self.agents)):
            self.agents[i]["agent"].snake.display = self.screen
            self.agents[i]["agent"].snake.font = self.font

        self.current_agent_idx = 0
        self.curr_steps = 0

    def get_current_agent(self):
        """
        Gets the current agent and updates the current agent idx
        Used to simulate each agent one by one
        """
        self.current_agent_idx += 1
        return self.agents[self.current_agent_idx-1]

    def update_agent(self):
        """
        Updates each agents values after a run/simulation
        """
        curr_agent = self.agents[self.current_agent_idx-1]
        curr_agent["steps"] = self.curr_steps
        curr_agent["fitness"] = curr_agent["agent"].fitness(self.curr_steps)
        curr_agent["score"] = curr_agent["agent"].snake.score

        self.curr_steps = 0

    def update_steps(self):
        """
        Updates the steps taken by snake 
        """
        self.curr_steps += 1

    def sort_agents(self) -> dict:
        """
        Sorts agents in descending order based on their fitness
        """
        return dict(
            sorted(
                self.agents.items(),
                key=lambda item: -item[1]["fitness"]
            )
        )

    def __combine_weights_s2(self, prev_m, new_m):
        """
        It combines the weights of the prev_m and new_m
        using some strategy

        Currently hard coded based on the model architecture
        """
        def change_weights(p, n):
            n = 0
            n = p + (np.random.randn(*p.shape) * np.random.randn(*p.shape))
            return n
        
        def change_biases(p, n):
            n = 0
            n = p + (0.1 * np.random.randn(*p.shape))
            return n
        
        # updating weights
        new_m.l1 = change_weights(prev_m.l1, new_m.l1)
        new_m.l2 = change_weights(prev_m.l2, new_m.l2)

        # updating biases
        new_m.l1_b = change_biases(prev_m.l1_b, new_m.l1_b)
        new_m.l2_b = change_biases(prev_m.l2_b, new_m.l2_b)

        return

    def __combine_agents_stragegy_2(self, agents):
        """
        Implements the strategy to combine agents from the current agents
        Here's how it works:
        - Pick top 10% of agents from the N agents
        - Drop the remaining agents i.e., the 90%
        - Make 90% new agents from the selected 10% agents.
        - Continue until end.
        """
        new_agents = dict()
        idx = 0

        # pick the top 10% agents
        N = len(agents)
        no_top_agents = int(0.1 * N)

        # this is cause we are going to change the items in new_agents,
        # and we cannot iterate over it in the same time.
        new_agents = dict()
        temp = dict()

        for k, v in agents.items():
            if no_top_agents == 0:
                break
            no_top_agents -= 1
            # print("the agent taken is: ")
            # print(f"{k}")
            # print(f"{v}")
            # print("\n")

            new_agents[idx] = v
            new_agents[idx]["fitness"] = 0
            new_agents[idx]["score"] = 0
            new_agents[idx]["steps"] = 0
            new_agents[idx]["gen"] = self.gen
            new_agents[idx]["agent_no"] = idx

            temp[idx] = v
            temp[idx]["fitness"] = 0
            temp[idx]["score"] = 0
            temp[idx]["gen"] = self.gen
            temp[idx]["steps"] = 0
            temp[idx]["agent_no"] = idx

            idx += 1

        # we are making 9 new agents from existing agents
        # The math works better
        for curr_agent in temp.values():
            for mutation_no in range(0, 9):
                try:
                    # create new agent
                    m = Agent(snake=Snake(w=self.w,
                                          h=self.h),
                              model=SnakeModel)
                    m1 = curr_agent["agent"].model

                    # print(f"new model weights inttial")
                    # print(m.model.state_dict())

                    # update the new agents weights
                    self.__combine_weights_s2(prev_m=m1,
                                              new_m=m.model)

                    # print(f"new model new weights")
                    # print(m.model.state_dict())

                    new_agents[idx] = {
                        "agent": m,
                        "fitness": 0,
                        "color": Color.GREEN.value,
                        "steps": 0,
                        "gen": self.gen,
                        "score": 0,
                        "agent_no": idx,
                    }
                    idx += 1

                except Exception as e:
                    # this is when there aren't two value or
                    # when len(agents) is odd
                    print(f"Some excpetion occured for: {idx}")
                    print(e)
                    pass

        return new_agents

    def __print_agents(self, agents):
        for k, v in agents.items():
            print(f"Agent no: {k}")
            print(f"Fitness:  {v['fitness']}")
            print(f"Score:    {v['score']}")
            print(f"Steps:    {v['steps']}")
            print(f"GEN:      {v['gen']}")
            print("\n")
        print("\n")

    def next_generation(self):
        """
        Take current agents (n) and mutate them into n/2 or n/2-1

        Steps:
        - sort the agents based on fitness.
        - make new agents using some strategy
        - make new agent from the combined weights
        - update the agents list
        """
        self.gen += 1
        # print(f"normal agents are \n")
        # print(f"nowmal len{len(self.agents)}")
        # self.__print_agents(self.agents)

        sorted_agents = self.sort_agents()

        # print(f"Sorted agents are")
        # print(f"sorlen {len(sorted_agents)}")
        # self.__print_agents(sorted_agents)

        # print(f"normal agents are \n")
        # self.__print_agents(self.agents)

        # taking average
        N = len(sorted_agents)
        self.this_gen_best_scores["avg_score"] = sum(
            x["score"] for x in sorted_agents.values()) / N
        self.this_gen_best_scores["avg_fitness"] = sum(
            x["fitness"] for x in sorted_agents.values()) / N

        # update best scores
        for k, v in sorted_agents.items():

            self.this_gen_best_scores["fitness"] = v["fitness"]
            self.this_gen_best_scores["score"] = v["score"]

            if self.best_scores["fitness"] < v["fitness"]:
                self.best_scores["fitness"] = v["fitness"]
                self.best_scores["best_fitness_agent"] = v["agent"]

            if self.best_scores["score"] < v["score"]:
                self.best_scores["score"] = v["score"]
                self.best_scores["best_score_agent"] = v["agent"]

            break

        # print("Best scores")
        # for k, v in self.best_scores.items():
        #     print(f"{k}: {v}")
        try:
            # print("let comgine noew")
            combined_agents = self.__combine_agents_stragegy_2(
                agents=sorted_agents)

            # print(f"combine agets len {len(combined_agents)}")
            # print("Combined agents \n")
            # self.__print_agents(combined_agents)
            # return

            self.agents = combined_agents

            # del sorted_agents
            # del combined_agents
            self.__initialize()
        except Exception as e:
            print("Error ", e)

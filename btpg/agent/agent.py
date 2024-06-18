
class Agent(object):
    env = None
    scene = None
    response_frequency = 1

    def __init__(self):
        self.condition_set = set()
        self.init_statistics()

    def bind_bt(self,bt):
        self.bt = bt
        bt.bind_agent(self)

    def init_statistics(self):
        self.step_num = 1
        self.next_response_time = self.response_frequency
        self.last_tick_output = None

    def step(self):
        if self.env.time > self.next_response_time:
            self.next_response_time += self.response_frequency
            self.step_num += 1

            self.bt.tick()
            bt_output = self.bt.visitor.output_str

            if bt_output != self.last_tick_output:
                if self.env.print_ticks:
                    print(f"==== time:{self.env.time:f}s ======")

                    # print(bt_output)
                    # 分割字符串
                    parts = bt_output.split("Action", 1)
                    # 获取 'Action' 后面的内容
                    if len(parts) > 1:
                        bt_output = parts[1].strip()  # 使用 strip() 方法去除可能的前后空格
                    else:
                        bt_output = ""  # 如果 'Action' 不存在于字符串中，则返回空字符串
                    print("Action ",bt_output)
                    print("\n")

                    self.last_tick_output = bt_output
                return True
            else:
                return False

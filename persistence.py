from langgraph.checkpoint.sqlite import SqliteSaver


class Persistence:
    @staticmethod
    def synchronous(conn_str: str = ":memory:"):
        cm = SqliteSaver.from_conn_string(conn_str)
        saver = cm.__enter__()
        saver._cm_ref = cm
        return saver


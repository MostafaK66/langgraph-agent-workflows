"""
persistence.py
Simple helper to create a *live* SqliteSaver.
"""

from langgraph.checkpoint.sqlite import SqliteSaver


class Persistence:
    @staticmethod
    def synchronous(conn_str: str = ":memory:"):
        """
        Return a live SqliteSaver object.

        Parameters
        ----------
        conn_str : str
            ":memory:"              → transient in-RAM DB
            "sqlite:///file.db"     → on-disk DB
        """
        cm = SqliteSaver.from_conn_string(conn_str)   # returns context-manager
        saver = cm.__enter__()                        # open connection
        # keep a reference to the cm so GC doesn't close it
        saver._cm_ref = cm                            # <-- key line
        return saver


import logging

from abc import ABC, abstractmethod
from pydantic import BaseModel

from openai_function_calling import FunctionInferrer

from spielberg.core.session import Session, OutputMessage

logger = logging.getLogger(__name__)


class AgentResult:
    SUCCESS = "success"
    ERROR = "error"


class AgentResponse(BaseModel):
    result: str = AgentResult.SUCCESS
    message: str = ""
    data: dict = {}


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, session: Session, **kwargs):
        self.session: Session = session
        self.output_message: OutputMessage = self.session.output_message

    def get_parameters(self):
        function_inferrer = FunctionInferrer.infer_from_function_reference(
            self.__call__
        )
        function_json = function_inferrer.to_json_schema()
        parameters = function_json.get("parameters")
        if not parameters:
            raise Exception(
                "Failed to infere parameters, please define JSON instead of using this automated util."
            )
        return parameters

    def to_llm_format(self):
        """Convert the agent to LLM tool format."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def name(self):
        return self.agent_name

    @property
    def agent_description(self):
        return self.description

    def safe_call(self, *args, **kwargs):
        try:
            return self.__call__(*args, **kwargs)

        except Exception as e:
            logger.exception(f"error in {self.agent_name} agent: {e}")
            return AgentResponse(result=AgentResult.ERROR, message=str(e))

    @abstractmethod
    def __call__(*args, **kwargs) -> AgentResponse:
        pass
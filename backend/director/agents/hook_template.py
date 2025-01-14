import logging
import json

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, ContextMessage, RoleTypes, MsgStatus, TextContent
from director.tools.videodb_tool import VideoDBTool
from director.llm import get_default_llm

logger = logging.getLogger(__name__)

class HookTemplateAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "hook_template"
        self.description = "Generates video hook templates by analyzing video content and creating structured recommendations."
        self.parameters = self.get_parameters()
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    @classmethod
    def get_parameters(cls):
        return {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "ID of the video to analyze"
                },
                "collection_id": {
                    "type": "string",
                    "description": "Collection ID of the video"
                },
                "prompt": {
                    "type": "string",
                    "description": "Description of the desired hook style"
                }
            },
            "required": ["video_id", "collection_id", "prompt"]
        }

    def _get_transcript(self, video_id):
        """Get video transcript."""
        self.output_message.actions.append("Processing video transcript...")
        self.output_message.push_update()
        try:
            transcript_text = self.videodb_tool.get_transcript(video_id, text=True)
            return transcript_text
        except Exception:
            self.output_message.actions.append("Indexing video speech...")
            self.output_message.push_update()
            self.videodb_tool.index_spoken_words(video_id)
            return self.videodb_tool.get_transcript(video_id, text=True)

    def _get_scenes(self, video_id):
        """Get video scenes, creating index if needed."""
        self.output_message.actions.append("Analyzing video scenes...")
        self.output_message.push_update()
        
        scene_list = self.videodb_tool.list_scene_index(video_id)
        if scene_list:
            scene_index_id = scene_list[0]["scene_index_id"]
            return self.videodb_tool.get_scene_index(video_id, scene_index_id)
        else:
            self.output_message.actions.append("Scene index not found. Creating scene index...")
            self.output_message.push_update()
            
            try:
                # Create scene index
                scene_index = self.videodb_tool.index_scene(video_id)
                if scene_index:
                    return self.videodb_tool.get_scene_index(video_id, scene_index["scene_index_id"])
                else:
                    logger.warning("Failed to create scene index")
                    return None
            except Exception as e:
                logger.warning(f"Error creating scene index: {str(e)}")
                return None

    def run(self, video_id: str, collection_id: str, prompt: str, *args, **kwargs) -> AgentResponse:
        """
        Generate a hook template based on video analysis.

        :param str video_id: The ID of the video to analyze
        :param str collection_id: The collection ID containing the video
        :param str prompt: User's description of desired hook style
        :return: The response containing the generated hook template
        :rtype: AgentResponse
        """
        try:
            # Initialize text content
            text_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Analyzing video content...",
            )
            self.output_message.content.append(text_content)
            self.output_message.push_update()

            # Initialize VideoDBTool and get content
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            
            # Get transcript and scenes
            transcript = self._get_transcript(video_id)
            scenes = self._get_scenes(video_id)

            # Create LLM prompt
            llm_prompt = f"""
            You are an expert video editor. Create an engaging hook template by analyzing this content carefully.
            Focus on capturing specific quotes and insights that will grab attention.

            User's Intent: {prompt}
            
            Content Analysis:
            1. Find the most attention-grabbing quote or insight from the first minute
            2. Identify the core message or surprising statement
            3. Note the exact visual setting and speaker dynamics
            4. Match the actual energy and style of the conversation
            
            Content to Analyze:
            Transcript: {transcript[:1000] if transcript else 'No transcript available'}
            Scenes: {json.dumps([scene['description'] for scene in scenes[:5]]) if scenes else 'No scenes available'}
            
            Create a JSON response with:
            {{
                "script": "Hook script that uses an actual quote or insight from the video",
                "visuals": [
                    "IMPORTANT: Only describe scenes that are explicitly mentioned in the transcript or scene descriptions.",
                    "Do not invent or add scenes that aren't in the source material.",
                    "Focus on the podcast studio setting and speaker interactions"
                ],
                "transitions": "Simple, clean transitions that match the professional studio setting",
                "music": "Subtle background music that won't overshadow the conversation",
                "sound_effects": "Minimal to none - only if absolutely necessary for the studio setting",
                "pacing": "Match the natural rhythm of the actual conversation"
            }}

            IMPORTANT GUIDELINES:
            1. Only use visual elements that are explicitly present in the video
            2. Avoid imagined or additional scenes not in the source material
            3. Keep the focus on the actual studio conversation
            4. Minimize unnecessary sound effects
            5. Maintain the authentic podcast atmosphere

            The hook script should incorporate real quotes or insights from the video to make it authentic and engaging.
            """

            # Get template from LLM
            self.output_message.actions.append("Generating hook template...")
            self.output_message.push_update()

            llm_response = self.llm.chat_completions(
                [ContextMessage(content=llm_prompt, role=RoleTypes.user).to_llm_msg()],
                response_format={"type": "json_object"},
            )

            if not llm_response.status:
                raise Exception("Failed to generate hook template")

            # Parse response and format template
            template_data = json.loads(llm_response.content)
            template = f"""
Hook Script:
"{template_data['script']}"

Visual Elements:
{chr(10).join(f'- {visual}' for visual in template_data['visuals'])}

Transitions:
{template_data['transitions']}

Audio:
- Music: {template_data['music']}
- Sound Effects: {template_data['sound_effects']}

Pacing:
{template_data['pacing']}
            """

            # Update output with result
            text_content.text = template
            text_content.status = MsgStatus.success
            text_content.status_message = "Hook template generated successfully"
            self.output_message.publish()

            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message=f"Agent {self.name} completed successfully.",
                data={"template": template}
            )

        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            if 'text_content' in locals():
                text_content.status = MsgStatus.error
                text_content.status_message = f"Error generating hook template: {str(e)}"
                self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Agent failed with error: {str(e)}"
            )
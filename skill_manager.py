import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

SKILL_PATH = "skill/skills.json"

class Skill(BaseModel):
    name: str = Field(
        description="Unique name of a skill, composed of Lowercase English letters and underlines. For example 'generate_codebase_readme'"
    )
    description: str = Field(
        description="A detailed description of the function and scenarios of the skill, which determines whether the LLM will choose to use it."
    )
    steps: List[str] = Field(
        description="A general step list of the skill. these steps must NOT contain any hard-coded data (such as particular file names), but should use pronouns or variables to indicate them."
    )

class SkillDecision(BaseModel):
    is_skill: bool = Field(
        description="Whether the task should be extracted as a reusable skill."
    )
    skill: Optional[Skill] = Field(
        default=None,
        description="Details about the extracted skill. if is_skill is false, set to None"
    )

class SkillManager:
    def __init__(self, file_path = SKILL_PATH):
        self.file_path = file_path
        self.skills: Dict[str, Skill] = self.load_skills()
    
    def load_skills(self) -> Dict[str, Skill]:
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as file:
                json.dump({}, file)
            return {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return {name: Skill(**skill_data) for name, skill_data in data.items()}
        except Exception as e:
            print(f"[SkillManager] Load failed: {e}")
            return {}

    def save_skill(self, skill: Skill) -> str:
        self.skills[skill.name] = skill
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json_data = {name: s.model_dump() for name, s in self.skills.items()}
            json.dump(json_data, file, ensure_ascii=False, indent=4)
        print(f"[SkillManager] skills updated. [{skill.name}] already saved")
    
    def get_skills_prompt(self) -> str:
        if not self.skills:
            return "You dont have any skill. If you encounter a complex problem, break it down into steps yourself."
        prompt = [""]
        for _, skill in self.skills.items():
            prompt.append(f"   - skill name: {skill.name}")
            prompt.append(f"     skill description: {skill.description}")
            prompt.append("   ->".join(skill.steps))
        return "\n".join(prompt)

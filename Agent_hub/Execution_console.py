# Entity/Agent_hub/Execution_console.py

import sys
import os
import re
import difflib
import ast
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pyautogui
import pytesseract
from PIL import Image, ImageGrab
import numpy as np

from Entity.Utils.Common_utilities import llm_call
from Entity.Memory.Operational_memory import OP_MEMORY
from Entity.Core.Component import BaseComponent

logger = logging.getLogger("Audio.agent")
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def agent_action(func):
    func.is_agent_action = True
    return func


class EXE:
    def __init__(self):
        self.notes: List[str] = []


class ACTIO(EXE):
    def __init__(self, platform: str, engine_parameters: Dict, width: int = 1920, height: int = 1080):
        super().__init__()
        self.platform = platform
        self.actual_width, self.actual_height = pyautogui.size()
        self.expected_width = engine_parameters.get("coordinate_width", self.actual_width)
        self.expected_height = engine_parameters.get("coordinate_height", self.actual_height)
        self.coords1: Optional[List[int]] = None
        self.coords2: Optional[List[int]] = None
        self.returned_info = None
        self.engine_parameters = engine_parameters
        self.safety_mode = os.getenv("AGENT_SAFETY_MODE", "confirm")
        self.destructive_actions = ["type", "hotkey", "press", "drag_and_drop"]
        self.last_destructive_action = None

        # Set pytesseract path from environment variable if provided
        tesseract_path = os.getenv("TESSERACT_PATH")
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract path set to: {tesseract_path}")
        else:
            logger.info("Using default tesseract path or system PATH")

        self.base_component = BaseComponent(engine_parameters, platform)
        try:
            self.ac_model = self.base_component._create_agent(
                "You are a computer vision assistant that identifies UI elements and returns coordinates. Respond ONLY with coordinates in format (x,y)."
            )
        except Exception as e:
            logger.warning("Could not create ac_model from BaseComponent: %s. Falling back to None.", e)
            self.ac_model = None

        try:
            text_agent_prompt = OP_MEMORY.PROMPT if hasattr(OP_MEMORY, 'PROMPT') else "You are a helpful assistant."
            self.text_span_agent = self.base_component._create_agent(text_agent_prompt)
        except Exception as e:
            logger.warning("Could not create text_span_agent from BaseComponent: %s. Falling back to None.", e)
            self.text_span_agent = None

        # Remove EasyOCR initialization and use pytesseract directly
        self.reader = None  # We'll use pytesseract directly

    def capture_screenshot(self) -> Image.Image:
        try:
            screenshot = ImageGrab.grab()
            return screenshot
        except Exception as e:
            logger.error("Failed to capture screenshot: %s", e)
            raise

    def _extract_text_with_pytesseract(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text and bounding boxes using pytesseract"""
        try:
            # Convert to RGB if needed (pytesseract works with RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Use pytesseract to get data with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Process the results
            results = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])

                # Filter out low confidence results and empty text
                if confidence > 30 and text:  # Lower confidence threshold for pytesseract
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]

                    # Calculate center coordinates
                    center_x = x + w // 2
                    center_y = y + h // 2

                    results.append({
                        'text': text,
                        'confidence': confidence / 100.0,  # Convert to 0-1 scale
                        'bbox': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                        'center': (center_x, center_y)
                    })

            return results

        except Exception as e:
            logger.error("pytesseract text extraction failed: %s", e)
            return []

    def _describe_screenshot_elements(self, screenshot: Image.Image) -> str:
        """Analyze screenshot and describe key UI elements for coordinate detection using pytesseract"""
        try:
            # Extract text elements using pytesseract
            ocr_results = self._extract_text_with_pytesseract(screenshot)

            # Extract prominent text elements
            prominent_elements = []
            for result in ocr_results[:15]:  # Limit to top 15 elements
                text = result['text']
                x, y = result['center']
                prominent_elements.append(f"'{text}' at ({x},{y})")

            # Get screen dimensions for context
            width, height = screenshot.size
            description = f"Screen: {width}x{height}. Visible elements: {', '.join(prominent_elements)}"

            return description

        except Exception as e:
            logger.debug("Screenshot description failed: %s", e)
            return "Unable to analyze screenshot"

    def gen_coordinates(self, ref_expr: str, screenshot: Optional[Image.Image] = None) -> List[int]:
        logger.debug("gen_coordinates called for ref_expr='%s'", ref_expr)
        if screenshot is None:
            try:
                screenshot = self.capture_screenshot()
            except Exception:
                screenshot = None

        # ENHANCED: Use visual context with LLM for accurate coordinate detection
        if self.ac_model is not None and screenshot is not None:
            try:
                self.ac_model.reset()

                # Get visual context from screenshot
                visual_context = self._describe_screenshot_elements(screenshot)

                visual_prompt = f"""
                Analyze this screenshot and locate: '{ref_expr}'
                Respond ONLY with coordinates: (x,y)

                Screenshot context: {visual_context}

                Important:
                - Return ONLY coordinates in format: (123,456)
                - No explanations, no other text
                - Use actual screen coordinates based on element positions
                - If uncertain, make your best estimate
                """

                self.ac_model.add_message(visual_prompt, role="user")
                response = llm_call(self.ac_model)
                logger.debug("Enhanced ac_model response for '%s': %s", ref_expr, response)

                # Parse coordinates from response
                numericals = re.findall(r"-?\d+", response)
                if len(numericals) >= 2:
                    coords = [int(numericals[0]), int(numericals[1])]
                    logger.debug("Parsed coordinates from enhanced ac_model: %s", coords)

                    # Validate coordinates are within screen bounds
                    if (0 <= coords[0] <= self.actual_width and
                            0 <= coords[1] <= self.actual_height):
                        return coords
                    else:
                        logger.warning("Coordinates out of bounds: %s", coords)

                patterns = [
                    r'\((\-?\d+),\s*(\-?\d+)\)',
                    r'\[(\-?\d+),\s*(\-?\d+)\]',
                    r'(\-?\d+)\s*,\s*(\-?\d+)',
                    r'x[:=]\s*(\-?\d+)\s*[,\s;]\s*y[:=]\s*(\-?\d+)'
                ]
                for pattern in patterns:
                    m = re.search(pattern, response)
                    if m:
                        coords = [int(m.group(1)), int(m.group(2))]
                        logger.debug("Parsed coordinates using pattern '%s': %s", pattern, coords)

                        # Validate bounds
                        if (0 <= coords[0] <= self.actual_width and
                                0 <= coords[1] <= self.actual_height):
                            return coords

                logger.debug("Enhanced ac_model did not return usable coordinates for '%s'", ref_expr)
            except Exception as e:
                logger.debug("Enhanced ac_model coordinate attempt failed: %s", e)

        # Fallback to original methods
        quick_match = re.search(r'(\-?\d+)\s*,\s*(\-?\d+)', ref_expr)
        if quick_match:
            coords = [int(quick_match.group(1)), int(quick_match.group(2))]
            logger.debug("Extracted coordinates directly from ref_expr: %s", coords)
            return coords

        # PYTESSERACT OCR fallback
        if screenshot is not None:
            try:
                ocr_results = self._extract_text_with_pytesseract(screenshot)
                logger.debug("pytesseract found %d text boxes", len(ocr_results))

                tokens = re.findall(r'\w+', ref_expr)
                if not tokens:
                    tokens = [ref_expr.strip()]

                best_score = 0.0
                best_coord = None

                for result in ocr_results:
                    text = result['text']
                    confidence = result['confidence']
                    x, y = result['center']

                    for token in tokens:
                        ratio = difflib.SequenceMatcher(None, token.lower(), text.lower()).ratio()
                        score = ratio * confidence
                        if score > best_score and score > 0.25:
                            best_score = score
                            best_coord = [x, y]

                if best_coord:
                    logger.debug("pytesseract matched token near '%s' -> %s (score %.3f)", ref_expr, best_coord,
                                 best_score)
                    return best_coord

            except Exception as e:
                logger.debug("pytesseract OCR fallback failed: %s", e)

        if self.text_span_agent is not None:
            try:
                self.text_span_agent.reset()
                prompt = f"I need coordinates for UI element described as: '{ref_expr}'. Provide 'x,y' if known, otherwise reply 'unknown'."
                self.text_span_agent.add_message(prompt, role="user")
                resp = llm_call(self.text_span_agent)
                numericals = re.findall(r"-?\d+", resp)
                if len(numericals) >= 2:
                    coords = [int(numericals[0]), int(numericals[1])]
                    logger.debug("text_span_agent provided coordinates: %s", coords)
                    return coords
            except Exception as e:
                logger.debug("text_span_agent attempt failed: %s", e)

        raise RuntimeError(f"Could not obtain coordinates for reference: '{ref_expr}'")

    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        if not coordinates or len(coordinates) < 2:
            return coordinates

        scaled = [
            round(coordinates[0] * self.actual_width / self.expected_width),
            round(coordinates[1] * self.actual_height / self.expected_height),
        ]
        logger.debug("Resized coords %s -> %s (screen %dx%d, expected %dx%d)",
                     coordinates, scaled, self.actual_width, self.actual_height,
                     self.expected_width, self.expected_height)
        return scaled

    def _require_confirmation(self, action_name: str, args: List, kwargs: Dict) -> bool:
        if self.safety_mode == "execute":
            return True
        elif self.safety_mode == "dry-run":
            logger.info(f"DRY-RUN: Would execute {action_name} with args {args} {kwargs}")
            return False

        action_str = f"{action_name}({', '.join(map(str, args))}"
        for k, v in kwargs.items():
            action_str += f", {k}={v}"
        action_str += ")"

        print(f"\n⚠️  DESTRUCTIVE ACTION REQUIRES CONFIRMATION:")
        print(f"Action: {action_str}")
        response = input("Type 'CONFIRM' to proceed or anything else to cancel: ")

        if response.strip().upper() == "CONFIRM":
            self.last_destructive_action = (action_name, args, kwargs)
            return True
        return False

    def assign_coordinates(self, plan_text: str, context: Dict):
        logger.info("assign_coordinates called with plan: %s", plan_text[:200])
        self.coords1, self.coords2 = None, None
        screenshot = context.get("screenshot") if context else None

        self.coords1 = None
        self.coords2 = None

        interactive_keywords = ["click", "type", "drag", "scroll", "press", "hotkey", "at", "on", "button"]
        if not any(keyword in plan_text.lower() for keyword in interactive_keywords):
            logger.info("No UI interaction keywords found, skipping coordinate detection")
            return

        strategies = [
            self._try_quoted_matches,
            self._try_single_matches,
            self._try_pytesseract_fallback
        ]

        for strat in strategies:
            try:
                logger.debug("Running coordinate strategy: %s", getattr(strat, "__name__", str(strat)))
                ok = strat(plan_text, screenshot)
                if ok:
                    logger.debug("Strategy %s succeeded. coords1=%s coords2=%s", strat.__name__, self.coords1,
                                 self.coords2)
                    return
            except Exception as e:
                logger.debug("Coordinate strategy %s failed: %s", getattr(strat, "__name__", str(strat)), e)
        logger.info("assign_coordinates completed: coords1=%s coords2=%s", self.coords1, self.coords2)
        if self.coords1 and 0 <= self.coords1[0] <= self.actual_width and 0 <= self.coords1[1] <= self.actual_height:
            return
        else:
            logger.warning("Invalid coordinates detected: %s", self.coords1)
            self.coords1, self.coords2 = None, None

    def _try_single_matches(self, plan_text: str, screenshot: Optional[Image.Image]) -> bool:
        # Look for specific UI element references, not generic text
        specific_patterns = [
            r"(?:click|select|press|type|at|on)\s+(?:the\s+)?([\"']?)(ok|cancel|save|open|close|submit|search|menu|file|edit|view|help)(?:\\1)(?:\s+button)?",
            r"(?:click|select|press)\s+(?:the\s+)?([\"']?)(start|windows|search|settings)(?:\\1)",
            r"([\"'])([^\"']+?)(?:\\1)\s+(?:button|icon|link|tab|menu)"
        ]

        for pattern in specific_patterns:
            matches = re.findall(pattern, plan_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    ref = matches[0][1]  # Get the actual text from the tuple
                else:
                    ref = matches[0]
                try:
                    self.coords1 = self.gen_coordinates(ref, screenshot)
                    if self.coords1:
                        return True
                except Exception as e:
                    logger.debug("Single-match coordinate generation failed for '%s': %s", ref, e)

        return False

    def _try_quoted_matches(self, plan_text: str, screenshot: Optional[Image.Image]) -> bool:
        quoted_matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', plan_text)
        if not quoted_matches:
            return False
        refs = [q[0] or q[1] for q in quoted_matches]
        for ref in refs:
            try:
                self.coords1 = self.gen_coordinates(ref.strip(), screenshot)
                if self.coords1:
                    return True
            except Exception:
                continue
        return False

    def _try_pytesseract_fallback(self, plan_text: str, screenshot: Optional[Image.Image]) -> bool:
        if not screenshot:
            return False
        try:
            ocr_results = self._extract_text_with_pytesseract(screenshot)
            words = re.findall(r'\b\w+\b', plan_text)
            if not words:
                return False
            for word in words:
                for result in ocr_results:
                    text = result['text']
                    confidence = result['confidence']
                    ratio = difflib.SequenceMatcher(None, word.lower(), text.lower()).ratio()
                    if ratio > 0.6 and confidence > 0.35:
                        x, y = result['center']
                        self.coords1 = [x, y]
                        logger.debug("pytesseract fallback matched '%s' -> %s (ocr_text=%s, conf=%.2f)", word,
                                     self.coords1,
                                     text, confidence)
                        return True
            return False
        except Exception as e:
            logger.debug("pytesseract fallback failed: %s", e)
            return False

    def _execute_and_return(self, action):
        if callable(action):
            try:
                result = action()
                if isinstance(result, (int, float)):
                    return str(result)
                elif isinstance(result, (list, tuple)):
                    return list(result)
                elif result is None:
                    return "Action completed successfully"
                else:
                    return result
            except Exception as e:
                logger.error("Execution failed: %s", e, exc_info=True)
                return f"Execution failed: {e}"
        logger.warning("Command blocked for safety: %s", action)
        return "Blocked unsafe command"

    def _analyze_visual_changes(self, before_screenshot: Image.Image, after_screenshot: Image.Image) -> Dict[str, Any]:
        """Analyze visual changes between before and after screenshots"""
        try:
            # Simple pixel difference analysis
            before_array = np.array(before_screenshot.convert("RGB"))
            after_array = np.array(after_screenshot.convert("RGB"))

            # Calculate pixel difference
            diff = np.abs(before_array.astype(float) - after_array.astype(float))
            mean_diff = np.mean(diff)

            # Basic change detection
            change_threshold = 10.0  # Adjust based on testing
            significant_change = mean_diff > change_threshold

            # OCR-based text change detection using pytesseract
            text_changes = self._detect_text_changes(before_screenshot, after_screenshot)

            return {
                "significant_change": significant_change,
                "mean_pixel_difference": float(mean_diff),
                "text_changes": text_changes,
                "action_effective": significant_change or len(text_changes) > 0
            }

        except Exception as e:
            logger.error("Visual change analysis failed: %s", e)
            return {
                "significant_change": False,
                "mean_pixel_difference": 0.0,
                "text_changes": [],
                "action_effective": False,
                "analysis_error": str(e)
            }

    def _detect_text_changes(self, before_screenshot: Image.Image, after_screenshot: Image.Image) -> List[str]:
        """Detect text changes between screenshots using pytesseract OCR"""
        try:
            before_texts = set()
            after_texts = set()

            # Extract text from before screenshot using pytesseract
            before_ocr = self._extract_text_with_pytesseract(before_screenshot)
            for result in before_ocr:
                if result['confidence'] > 0.6 and len(result['text'].strip()) > 1:
                    before_texts.add(result['text'].strip().lower())

            # Extract text from after screenshot using pytesseract
            after_ocr = self._extract_text_with_pytesseract(after_screenshot)
            for result in after_ocr:
                if result['confidence'] > 0.6 and len(result['text'].strip()) > 1:
                    after_texts.add(result['text'].strip().lower())

            # Find new text that appeared
            new_texts = after_texts - before_texts
            # Find text that disappeared
            removed_texts = before_texts - after_texts

            changes = []
            if new_texts:
                changes.append(f"New text: {', '.join(list(new_texts)[:3])}")
            if removed_texts:
                changes.append(f"Removed text: {', '.join(list(removed_texts)[:3])}")

            return changes

        except Exception as e:
            logger.debug("Text change detection failed: %s", e)
            return []

    @agent_action
    def click(self, num_clicks: int = 1, button_type: str = "left", hold_keys: Optional[List[str]] = None):
        if self.coords1 is None:
            raise RuntimeError("coords1 not set before click action")
        if hold_keys is None:
            hold_keys = []
        x, y = self.resize_coordinates(self.coords1)

        def do_click():
            for k in hold_keys:
                pyautogui.keyDown(k)
            pyautogui.click(x, y, clicks=num_clicks, button=button_type)
            for k in hold_keys:
                pyautogui.keyUp(k)

        return self._execute_and_return(do_click)

    @agent_action
    def type(self, text: str = "", overwrite: bool = False, enter: bool = False):
        if not self._require_confirmation("type", [text], {"overwrite": overwrite, "enter": enter}):
            return "Action cancelled by safety interlock"

        def do_type():
            if self.coords1:
                x, y = self.resize_coordinates(self.coords1)
                pyautogui.click(x, y)
            if overwrite:
                pyautogui.hotkey('ctrl', 'a')
                pyautogui.press('backspace')
            pyautogui.write(str(text))
            if enter:
                pyautogui.press('enter')

        return self._execute_and_return(do_type)

    @agent_action
    def drag_and_drop(self, hold_keys: Optional[List[str]] = None):
        if not self._require_confirmation("drag_and_drop", [], {"hold_keys": hold_keys}):
            return "Action cancelled by safety interlock"

        if self.coords1 is None or self.coords2 is None:
            raise RuntimeError("coords1/coords2 not set before drag_and_drop action")
        if hold_keys is None:
            hold_keys = []
        x1, y1 = self.resize_coordinates(self.coords1)
        x2, y2 = self.resize_coordinates(self.coords2)

        def do_drag():
            for k in hold_keys:
                pyautogui.keyDown(k)
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=0.5, button='left')
            pyautogui.mouseUp()
            for k in hold_keys:
                pyautogui.keyUp(k)

        return self._execute_and_return(do_drag)

    @agent_action
    def highlight_text_span(self, button: str = "left"):
        if self.coords1 is None or self.coords2 is None:
            raise RuntimeError("coords1/coords2 not set before highlight_text_span action")
        x1, y1 = self.resize_coordinates(self.coords1)
        x2, y2 = self.resize_coordinates(self.coords2)

        def do_highlight():
            pyautogui.moveTo(x1, y1)
            pyautogui.mouseDown(button=button)
            pyautogui.moveTo(x2, y2, duration=0.5)
            pyautogui.mouseUp(button=button)

        return self._execute_and_return(do_highlight)

    @agent_action
    def open(self, app_or_filename: str):
        def do_open():
            try:
                if self.platform == "windows":
                    pyautogui.hotkey('win')
                    time.sleep(0.5)
                    pyautogui.write(app_or_filename)
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(2.0)
                elif self.platform == "macos":
                    pyautogui.hotkey('command', 'space')
                    time.sleep(0.5)
                    pyautogui.write(app_or_filename)
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(2.0)
                else:
                    pyautogui.hotkey('alt', 'f2')
                    time.sleep(0.5)
                    pyautogui.write(app_or_filename)
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(2.0)
                return f"Attempted to open: {app_or_filename}"
            except Exception as e:
                raise RuntimeError(f"open action failed: {e}")

        return self._execute_and_return(do_open)

    @agent_action
    def scroll(self, clicks: int, shift: bool = False):
        if self.coords1 is None:
            raise RuntimeError("coords1 not set before scroll action")
        x, y = self.resize_coordinates(self.coords1)

        def do_scroll():
            pyautogui.moveTo(x, y)
            time.sleep(0.2)
            if shift:
                pyautogui.hscroll(clicks)
            else:
                pyautogui.vscroll(clicks)

        return self._execute_and_return(do_scroll)

    @agent_action
    def hotkey(self, keys: List[str]):
        if not self._require_confirmation("hotkey", [], {"keys": keys}):
            return "Action cancelled by safety interlock"

        def do_hotkey():
            pyautogui.hotkey(*keys)

        return self._execute_and_return(do_hotkey)

    @agent_action
    def hold_and_press(self, hold_keys: List[str], press_keys: List[str]):
        if not self._require_confirmation("hold_and_press", [], {"hold_keys": hold_keys, "press_keys": press_keys}):
            return "Action cancelled by safety interlock"

        if hold_keys is None:
            hold_keys = []
        if press_keys is None:
            press_keys = []

        def do_hold_press():
            for k in hold_keys:
                pyautogui.keyDown(k)
            for k in press_keys:
                pyautogui.press(k)
            for k in hold_keys:
                pyautogui.keyUp(k)

        return self._execute_and_return(do_hold_press)

    @agent_action
    def wait(self, time_s: float):
        def do_wait():
            time.sleep(float(time_s))

        return self._execute_and_return(do_wait)

    @agent_action
    def done(self, return_value: Optional[Union[Dict, str, List, Tuple, int, float, bool]] = None):
        self.returned_info = return_value
        return "DONE"

    @agent_action
    def fail(self):
        return "FAIL"

    @agent_action
    def press(self, key: str):
        if not self._require_confirmation("press", [], {"key": key}):
            return "Action cancelled by safety interlock"

        def do_press():
            pyautogui.press(key)

        return self._execute_and_return(do_press)

    @agent_action
    def move_to(self, x: Optional[int] = None, y: Optional[int] = None):
        if x is not None and y is not None:
            target_x, target_y = x, y
        elif self.coords1:
            target_x, target_y = self.resize_coordinates(self.coords1)
        else:
            raise RuntimeError("No coordinates provided for move_to action")

        def do_move():
            pyautogui.moveTo(target_x, target_y)

        return self._execute_and_return(do_move)

    def _sanitize_code(self, code: str) -> str:
        if code is None:
            return ""
        sanitized = code.strip()
        sanitized = re.sub(r"^```(?:\w+)?\s*", "", sanitized)
        sanitized = re.sub(r"\s*```$", "", sanitized)
        sanitized = sanitized.strip("`")
        return sanitized.strip()

    def _extract_first_agent_function(self, code: str) -> Optional[str]:
        m = re.search(r'(?:agent|self)\.([A-Za-z_]\w*)\s*\(', code)
        if m:
            return m.group(1)
        return None

    def _validate_method_call(self, code: str, method_name: str) -> bool:
        patterns = [
            rf"(?:agent|self)\.{re.escape(method_name)}\s*\((.*?)\)",
            rf"(?:agent|self)\.{re.escape(method_name)}\s*\s*\(\s*\)"
        ]
        return any(re.search(p, code, re.DOTALL) for p in patterns)

    def _parse_call_args_kwargs(self, call_string: str) -> Tuple[List[Any], Dict[str, Any]]:
        call_string = call_string.strip()
        if not call_string:
            return [], {}
        try:
            fake = f"f({call_string})"
            tree = ast.parse(fake, mode="eval")
            if isinstance(tree, ast.Expression) and isinstance(tree.body, ast.Call):
                args = []
                kwargs = {}
                for a in tree.body.args:
                    try:
                        val = ast.literal_eval(a)
                    except Exception:
                        val = ast.unparse(a) if hasattr(ast, "unparse") else None
                    args.append(val)
                for k in tree.body.keywords:
                    key = k.arg
                    try:
                        val = ast.literal_eval(k.value)
                    except Exception:
                        val = ast.unparse(k.value) if hasattr(ast, "unparse") else None
                    kwargs[key] = val
                return args, kwargs
        except Exception:
            logger.debug("AST parsing of call args failed for: %s", call_string)
        parts = [p.strip() for p in re.split(r',(?![^\(\[]*[\]\)])', call_string) if p.strip()]
        args = []
        kwargs = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip()
                v = v.strip()
                try:
                    kwargs[k] = ast.literal_eval(v)
                except Exception:
                    kwargs[k] = v.strip('"\'')
            else:
                try:
                    args.append(ast.literal_eval(p))
                except Exception:
                    args.append(p.strip('"\''))
        return args, kwargs

    def _safe_execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]):
        if not hasattr(self, method_name):
            raise AttributeError(f"Agent does not have method '{method_name}'")
        method = getattr(self, method_name)
        if not getattr(method, "is_agent_action", False):
            raise PermissionError(
                f"Method '{method_name}' is not decorated with @agent_action and cannot be invoked by action_generator")
        logger.info("Executing agent method '%s' with args=%s kwargs=%s", method_name, args, kwargs)
        return method(*args, **kwargs)

    def execute_planned_step(self, step_description: str, action_type: str, target: str = None,
                             parameters: Dict = None) -> Tuple[Dict[str, Any], str]:
        """
        Execute a single planned step from Action Board with enhanced error handling

        Args:
            step_description: Description of what this step should accomplish
            action_type: Type of action to execute (click, type, open, etc.)
            target: Target element or application
            parameters: Additional parameters for the action

        Returns:
            Tuple[Dict, str]: Execution metadata and result
        """
        logger.info(f"Executing planned step: {action_type} - {step_description}")
        max_retries = parameters.get("max_retries", 3) if parameters else 3
        retry_delay = parameters.get("retry_delay", 1.0) if parameters else 1.0
        
        # Initialize execution metadata
        meta = {
            "step_description": step_description,
            "action_type": action_type,
            "target": target,
            "parameters": parameters,
            "coords1": None,
            "coords2": None,
            "visual_feedback": {},
            "retry_count": 0,
            "success": False,
            "errors": []
        }

        # Attempt execution with retries
        for attempt in range(max_retries):
            try:
                # Capture BEFORE screenshot for action verification
                try:
                    before_screenshot = self.capture_screenshot()
                    logger.debug("Before screenshot captured")
                except Exception as e:
                    logger.warning("Could not capture before screenshot: %s", e)
                    before_screenshot = None

                # Set coordinates if target is provided
                if target and action_type in ["click_element", "type_text", "scroll", "move_to"]:
                    try:
                        self.assign_coordinates(f"{action_type} {target}", {"screenshot": before_screenshot})
                        meta["coords1"] = self.coords1
                        meta["coords2"] = self.coords2
                    except Exception as e:
                        error_msg = f"Coordinate assignment failed: {e}"
                        logger.warning(error_msg)
                        meta["errors"].append(error_msg)
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            return meta, f"Coordinate assignment failed after {max_retries} attempts"

                # Execute the action based on type
                exec_result = None
                try:
                    if action_type == "click_element":
                        exec_result = self.click()

                    elif action_type == "type_text":
                        text = parameters.get("text", "") if parameters else ""
                        enter = parameters.get("enter", False) if parameters else False
                        exec_result = self.type(text=text, enter=enter)

                    elif action_type == "open_app":
                        exec_result = self.open(target)

                    elif action_type == "press_key":
                        exec_result = self.press(target)

                    elif action_type == "hotkey":
                        keys = target.split("+") if target else []
                        exec_result = self.hotkey(keys)

                    elif action_type == "scroll":
                        clicks = parameters.get("clicks", 1) if parameters else 1
                        direction = parameters.get("direction", "down") if parameters else "down"
                        shift = (direction == "horizontal")
                        exec_result = self.scroll(clicks=clicks, shift=shift)

                    elif action_type == "wait":
                        seconds = parameters.get("seconds", 1.0) if parameters else 1.0
                        exec_result = self.wait(seconds)

                    elif action_type == "done":
                        exec_result = self.done()

                    else:
                        logger.warning(f"Unknown action type: {action_type}, defaulting to wait")
                        exec_result = self.wait(1.0)

                except Exception as e:
                    error_msg = f"Action execution failed: {e}"
                    logger.error(error_msg, exc_info=True)
                    meta["errors"].append(error_msg)
                    exec_result = f"Execution failed: {e}"
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying step (attempt {attempt + 2}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        meta["success"] = False
                        return meta, exec_result

                # Capture AFTER screenshot and analyze visual changes
                visual_feedback = {}
                if before_screenshot is not None:
                    try:
                        after_screenshot = self.capture_screenshot()
                        visual_feedback = self._analyze_visual_changes(before_screenshot, after_screenshot)
                        logger.info("Visual feedback analysis: %s", visual_feedback)
                    except Exception as e:
                        error_msg = f"Visual change analysis failed: {e}"
                        logger.warning(error_msg)
                        visual_feedback = {"analysis_error": str(e)}
                        meta["errors"].append(error_msg)

                # Update metadata
                meta["visual_feedback"] = visual_feedback
                meta["retry_count"] = attempt
                meta["success"] = (
                    "fail" not in str(exec_result).lower() and 
                    "error" not in str(exec_result).lower() and
                    visual_feedback.get("action_effective", False)
                )

                logger.info(f"Step execution completed: {action_type} -> {exec_result}")
                return meta, exec_result
                
            except Exception as e:
                error_msg = f"Unexpected error in step execution: {e}"
                logger.error(error_msg, exc_info=True)
                meta["errors"].append(error_msg)
                meta["retry_count"] = attempt
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying due to unexpected error (attempt {attempt + 2}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    meta["success"] = False
                    return meta, f"Step failed after {max_retries} attempts: {e}"

        # This should never be reached due to the return statements above
        meta["success"] = False
        return meta, "Step execution failed unexpectedly"

    # Legacy method for backward compatibility
    def action_generator(self, instruction: str, iteration_context: Dict = None) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Legacy method - now executes planned steps from Action Board

        This is called by Agent1.py in the iterative loop to execute planned steps
        """
        logger.info(f"Executing planned step for: {instruction}")

        # Extract step info from iteration context
        step_description = instruction
        action_type = iteration_context.get("action_type", "wait") if iteration_context else "wait"
        target = iteration_context.get("target")
        parameters = iteration_context.get("parameters", {})

        # Execute the planned step
        meta, result = self.execute_planned_step(step_description, action_type, target, parameters)

        # Return format expected by Agent1.py
        return meta, [result]


# Global instance for compatibility
EXECUTION_CONSOLE = None


def get_execution_console(platform: str, engine_parameters: Dict, width: int = 1920, height: int = 1080) -> ACTIO:
    global EXECUTION_CONSOLE
    if EXECUTION_CONSOLE is None:
        EXECUTION_CONSOLE = ACTIO(platform, engine_parameters, width, height)
    return EXECUTION_CONSOLE
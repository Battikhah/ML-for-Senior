/** Validate the four-slide native deck before writing the professor's final editable file. */
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const reviewDir=path.resolve(process.argv[2]);
const workspaceDir=path.resolve(reviewDir,'../..');
const skill=process.env.PRESENTATION_SKILL_DIR;
if(!skill || !process.env.RUNTIME_PYTHON)throw Error('Supply the documented presentation skill/runtime paths.');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=path.join(reviewDir,'TreeVibes_Professor_Quick_Review_2026-10-20.pptx');
const result=await finalizePresentation({
  workspaceDir,candidatePath:path.join(reviewDir,'.deck-build/candidate.pptx'),finalPath,
  explicitTotalSlideCount:4,requiredNativeTableOwnerSlides:[2,3,4],
  requiredNativeChartOwnerSlides:[1,4],materializeLiteralChartWorkbooks:true,
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,3,4].flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(workspaceDir,'artifacts/validation/professor_review_verified_local.validation.json')
});
console.log(JSON.stringify(result));

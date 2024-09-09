// import { app } from "../../../scripts/app.js";
// import { ComfyWidgets } from "../../../scripts/widgets.js";
//
// // Displays input text on a node
// app.registerExtension({
// 	name: "toonsquare.WallpaperPromptGenerator",
// 	async beforeRegisterNodeDef(nodeType, nodeData, app) {
// 		if (nodeData.name === "Wallpaper Prompt Generator") {
// 			const onExecuted = nodeType.prototype.onExecuted;
// 			nodeType.prototype.onExecuted = function (message) {
// 				const r = onExecuted?.apply?.(this, arguments);
//
// 				const pos = this.widgets.findIndex((w) => w.name === "prompt");
// 				if (pos !== -1) {
// 					for (let i = pos; i < this.widgets.length; i++) {
// 						this.widgets[i].onRemove?.();
// 					}
// 					this.widgets.length = pos;
// 				}
// 				console.log(message);
// 				// for (const list of message.tags) {
// 				// 	const w = ComfyWidgets["STRING"](this, "tags", ["STRING", { multiline: true }], app).widget;
// 				// 	w.inputEl.readOnly = true;
// 				// 	w.inputEl.style.opacity = 0.6;
// 				// 	w.value = list;
// 				// }
// 				const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
// 				w.inputEl.readOnly = true;
// 				w.inputEl.style.opacity = 0.6;
// 				w.value = message.prompt;
//
// 				this.onResize?.(this.size);
//
// 				return r;
// 			};
// 		}
// 	},
// });

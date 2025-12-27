const bundledThemesInfo = [
];
const bundledThemes = Object.fromEntries(bundledThemesInfo.map((i) => [i.id, i.import]));

export { bundledThemes as b };
